#!/usr/bin/env python3
"""kv_router — cache-aware, optionally prefill/decode-disaggregated router for vLLM.

Why this exists
---------------
The blueprint points the SGLang router at the *decode* pool only, which means
the prefill pods never receive traffic, so config C would not actually be
disaggregated. This router adds the missing orchestration step:

  client ──► kv_router ──(1) prefill copy: max_tokens=1 ──► vllm-prefill-N
                 │                                           │ LMCache writes the
                 │                                           ▼ prompt KV to Mooncake
                 └────(2) original request, streamed ──► vllm-decode-M
                                                             │ LMCache reads the KV from
                                                             ▼ HBM / CPU / Mooncake
                                                         tokens stream back

With no prefill workers configured it is a plain cache-aware router (a drop-in
alternative to sglang_router for config B).

Cache-aware policy (same idea as SGLang's): the request text is cut into
fixed-size character blocks and hashed as a chain. Each worker remembers (LRU)
the chain hashes it has served. A request goes to the worker with the longest
matching prefix, unless the pool is imbalanced
(max_inflight - min_inflight > BALANCE_ABS and max > BALANCE_REL * min),
in which case it goes to the least-loaded worker.

Admission queue: with MAX_INFLIGHT_PER_WORKER > 0 each decode worker takes at
most that many requests; the rest wait in one bounded FIFO here (QUEUE_MAX,
QUEUE_TIMEOUT) instead of piling up in a single vLLM pod's waiting queue. The
worker is chosen when a slot frees (late binding), so a burst is spread over
whichever pods drain first. A full queue answers 429, a timed-out wait 503,
both with Retry-After. MAX_INFLIGHT_PER_WORKER=0 turns it off.

Configuration comes from env vars (see `Settings`). Only depends on aiohttp.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

log = logging.getLogger("kv_router")


def _urls(value: str) -> list[str]:
    return [u.strip().rstrip("/") for u in value.replace(",", " ").split() if u.strip()]


@dataclass
class Settings:
    decode_urls: list[str] = field(default_factory=lambda: _urls(os.environ.get("DECODE_URLS", "")))
    prefill_urls: list[str] = field(default_factory=lambda: _urls(os.environ.get("PREFILL_URLS", "")))
    policy: str = os.environ.get("POLICY", "cache_aware")          # cache_aware | round_robin | random
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8080"))
    metrics_port: int = int(os.environ.get("METRICS_PORT", "9100"))
    block_chars: int = int(os.environ.get("BLOCK_CHARS", "256"))
    max_blocks: int = int(os.environ.get("MAX_BLOCKS", "64"))        # only the first 64*256 chars matter
    lru_blocks: int = int(os.environ.get("LRU_BLOCKS", "200000"))    # per worker
    balance_abs: int = int(os.environ.get("BALANCE_ABS", "8"))
    balance_rel: float = float(os.environ.get("BALANCE_REL", "1.5"))
    health_interval: float = float(os.environ.get("HEALTH_INTERVAL", "5"))
    request_timeout: float = float(os.environ.get("REQUEST_TIMEOUT", "900"))
    max_inflight: int = int(os.environ.get("MAX_INFLIGHT_PER_WORKER", "0"))  # 0 = no cap, no queue
    queue_max: int = int(os.environ.get("QUEUE_MAX", "512"))
    queue_timeout: float = float(os.environ.get("QUEUE_TIMEOUT", "120"))


# ----------------------------------------------------------------------------- metrics
class Metrics:
    def __init__(self) -> None:
        self.counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self.help = {
            "kv_router_requests_total": "Requests routed, by pool and worker",
            "kv_router_errors_total": "Upstream errors, by pool and worker",
            "kv_router_prefix_matched_blocks_total": "Prefix blocks that matched the chosen worker's history",
            "kv_router_prefix_query_blocks_total": "Prefix blocks looked up",
            "kv_router_prefill_seconds_sum": "Total time spent in the prefill hop",
            "kv_router_prefill_seconds_count": "Number of prefill hops",
            "kv_router_rejected_total": "Requests turned away by the admission queue, by reason",
            "kv_router_queue_wait_seconds_sum": "Total time requests waited in the admission queue",
            "kv_router_queue_wait_seconds_count": "Number of requests that went through the admission queue",
        }

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        self.counters[(name, tuple(sorted(labels.items())))] += value

    def render(self, pools: list["Pool"], queue_depth: int = 0) -> str:
        out, seen = [], set()
        for (name, labels), val in sorted(self.counters.items()):
            if name not in seen:
                seen.add(name)
                kind = "counter" if name.endswith("_total") else "untyped"
                out += [f"# HELP {name} {self.help.get(name, name)}", f"# TYPE {name} {kind}"]
            lbl = ",".join(f'{k}="{v}"' for k, v in labels)
            out.append(f"{name}{{{lbl}}} {val}" if lbl else f"{name} {val}")
        out += ["# HELP kv_router_inflight In-flight requests per worker", "# TYPE kv_router_inflight gauge"]
        out += ["# HELP kv_router_worker_healthy 1 if the worker's /health is OK", "# TYPE kv_router_worker_healthy gauge"]
        for p in pools:
            for w in p.workers:
                out.append(f'kv_router_inflight{{pool="{p.name}",worker="{w.url}"}} {w.inflight}')
                out.append(f'kv_router_worker_healthy{{pool="{p.name}",worker="{w.url}"}} {int(w.healthy)}')
        out += ["# HELP kv_router_queue_depth Requests waiting in the admission queue",
                "# TYPE kv_router_queue_depth gauge", f"kv_router_queue_depth {queue_depth}"]
        return "\n".join(out) + "\n"


METRICS = Metrics()


# ----------------------------------------------------------------------------- routing
def prefix_hashes(text: str, block_chars: int, max_blocks: int) -> list[bytes]:
    """Chained hashes of the full blocks of `text` (hash i covers blocks 0..i)."""
    h = hashlib.blake2b(digest_size=8)
    out = []
    limit = min(len(text) // block_chars, max_blocks)
    for i in range(limit):
        h.update(text[i * block_chars:(i + 1) * block_chars].encode("utf-8", "ignore"))
        out.append(h.digest())
    return out


class Worker:
    def __init__(self, url: str, lru_blocks: int) -> None:
        self.url = url
        self.inflight = 0
        self.healthy = True
        self._blocks: OrderedDict[bytes, None] = OrderedDict()
        self._lru = lru_blocks

    def match(self, hashes: list[bytes]) -> int:
        n = 0
        for h in hashes:
            if h not in self._blocks:
                break
            n += 1
        return n

    def remember(self, hashes: list[bytes]) -> None:
        for h in hashes:
            self._blocks[h] = None
            self._blocks.move_to_end(h)
        while len(self._blocks) > self._lru:
            self._blocks.popitem(last=False)


class Pool:
    def __init__(self, name: str, urls: list[str], s: Settings) -> None:
        self.name = name
        self.s = s
        self.workers = [Worker(u, s.lru_blocks) for u in urls]
        self._rr = 0

    def candidates(self) -> list[Worker]:
        return [w for w in self.workers if w.healthy] or self.workers

    def pick(self, hashes: list[bytes], cands: list[Worker] | None = None) -> tuple[Worker, int]:
        cands = cands or self.candidates()
        if self.s.policy == "round_robin":
            self._rr += 1
            w = cands[self._rr % len(cands)]
        elif self.s.policy == "random":
            w = random.choice(cands)
        else:
            loads = [w.inflight for w in cands]
            lo, hi = min(loads), max(loads)
            imbalanced = hi - lo > self.s.balance_abs and hi > self.s.balance_rel * lo
            if imbalanced or not hashes:
                w = min(cands, key=lambda x: (x.inflight, random.random()))
            else:
                w = max(cands, key=lambda x: (x.match(hashes), -x.inflight, random.random()))
        matched = w.match(hashes)
        w.remember(hashes)
        METRICS.inc("kv_router_prefix_matched_blocks_total", matched, pool=self.name)
        METRICS.inc("kv_router_prefix_query_blocks_total", len(hashes), pool=self.name)
        return w, matched


class QueueFull(Exception):
    pass


class Admission:
    """Bounded FIFO in front of a pool. acquire() returns a worker whose
    `inflight` has already been incremented; the caller must release() it."""

    def __init__(self, pool: Pool, s: Settings) -> None:
        self.pool = pool
        self.s = s
        self.waiters: deque[tuple[list[bytes], asyncio.Future]] = deque()

    def _free(self) -> list[Worker]:
        cands = self.pool.candidates()
        if self.s.max_inflight <= 0:
            return cands
        return [w for w in cands if w.inflight < self.s.max_inflight]

    def _assign(self, free: list[Worker], hashes: list[bytes]) -> Worker:
        w, _ = self.pool.pick(hashes, free)
        w.inflight += 1
        return w

    async def acquire(self, hashes: list[bytes]) -> Worker:
        if not self.waiters:
            free = self._free()
            if free:
                return self._assign(free, hashes)
        if len(self.waiters) >= self.s.queue_max:
            METRICS.inc("kv_router_rejected_total", reason="queue_full")
            raise QueueFull
        fut = asyncio.get_running_loop().create_future()
        entry = (hashes, fut)
        self.waiters.append(entry)
        t0 = time.perf_counter()
        try:
            return await asyncio.wait_for(fut, self.s.queue_timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            if fut.done() and not fut.cancelled():
                self.release(fut.result())  # a slot was handed over just as we gave up
            try:
                self.waiters.remove(entry)
            except ValueError:
                pass
            if isinstance(e, asyncio.TimeoutError):
                METRICS.inc("kv_router_rejected_total", reason="queue_timeout")
            raise
        finally:
            METRICS.inc("kv_router_queue_wait_seconds_sum", time.perf_counter() - t0)
            METRICS.inc("kv_router_queue_wait_seconds_count")

    def release(self, w: Worker) -> None:
        w.inflight -= 1
        self.drain()

    def drain(self) -> None:
        while self.waiters:
            free = self._free()
            if not free:
                return
            hashes, fut = self.waiters.popleft()
            if not fut.done():
                fut.set_result(self._assign(free, hashes))


def request_text(body: dict) -> str:
    """The text the backend will see as the prompt prefix (approximately)."""
    if "messages" in body:
        parts = []
        for m in body.get("messages") or []:
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(f"<{m.get('role', '')}>{content or ''}")
        return "".join(parts)
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt and isinstance(prompt[0], str) else json.dumps(prompt)
    return str(prompt)


# ----------------------------------------------------------------------------- server
class Router:
    def __init__(self, s: Settings) -> None:
        if not s.decode_urls:
            raise SystemExit("DECODE_URLS is empty")
        self.s = s
        self.decode = Pool("decode", s.decode_urls, s)
        self.prefill = Pool("prefill", s.prefill_urls, s) if s.prefill_urls else None
        self.admission = Admission(self.decode, s)
        self.session: ClientSession | None = None

    @property
    def pools(self) -> list[Pool]:
        return [p for p in (self.prefill, self.decode) if p]

    async def start(self, app: web.Application) -> None:
        self.session = ClientSession(
            connector=TCPConnector(limit=0, ttl_dns_cache=30),
            timeout=ClientTimeout(total=None, sock_connect=10, sock_read=self.s.request_timeout),
        )
        app["health_task"] = asyncio.create_task(self._health_loop())

    async def stop(self, app: web.Application) -> None:
        app["health_task"].cancel()
        if self.session:
            await self.session.close()

    async def _health_loop(self) -> None:
        while True:
            for pool in self.pools:
                for w in pool.workers:
                    try:
                        async with self.session.get(f"{w.url}/health", timeout=ClientTimeout(total=3)) as r:
                            ok = r.status == 200
                    except Exception:
                        ok = False
                    if ok != w.healthy:
                        log.warning("worker %s (%s) healthy=%s", w.url, pool.name, ok)
                    w.healthy = ok
            self.admission.drain()  # a worker that came back may free queued requests
            await asyncio.sleep(self.s.health_interval)

    # -- handlers
    async def health(self, _: web.Request) -> web.Response:
        ok = any(w.healthy for w in self.decode.workers)
        return web.json_response({"status": "ok" if ok else "no healthy decode workers"}, status=200 if ok else 503)

    async def metrics(self, _: web.Request) -> web.Response:
        return web.Response(text=METRICS.render(self.pools, len(self.admission.waiters)),
                            content_type="text/plain")

    async def models(self, _: web.Request) -> web.Response:
        w = next((w for w in self.decode.workers if w.healthy), self.decode.workers[0])
        async with self.session.get(f"{w.url}/v1/models") as r:
            return web.Response(body=await r.read(), status=r.status, content_type="application/json")

    async def workers(self, _: web.Request) -> web.Response:
        out = {p.name: [{"url": w.url, "healthy": w.healthy, "inflight": w.inflight}
                        for w in p.workers] for p in self.pools}
        out["queue_depth"] = len(self.admission.waiters)
        return web.json_response(out)

    async def _prefill(self, path: str, body: dict, hashes: list[bytes]) -> None:
        """Run the prompt through a prefill worker so its KV lands in the shared
        store. Failures are non-fatal: decode will just compute the prefill itself."""
        w, _ = self.prefill.pick(hashes)
        pbody = dict(body)
        pbody["max_tokens"] = 1
        if "max_completion_tokens" in pbody:
            pbody["max_completion_tokens"] = 1
        pbody["stream"] = False
        for k in ("stream_options", "min_tokens", "n", "best_of"):
            pbody.pop(k, None)
        w.inflight += 1
        METRICS.inc("kv_router_requests_total", pool="prefill", worker=w.url)
        t0 = time.perf_counter()
        try:
            async with self.session.post(f"{w.url}{path}", json=pbody) as r:
                await r.read()
                if r.status != 200:
                    METRICS.inc("kv_router_errors_total", pool="prefill", worker=w.url)
                    log.warning("prefill %s -> HTTP %s", w.url, r.status)
        except Exception as e:  # noqa: BLE001
            METRICS.inc("kv_router_errors_total", pool="prefill", worker=w.url)
            log.warning("prefill %s failed: %s", w.url, e)
        finally:
            w.inflight -= 1
            METRICS.inc("kv_router_prefill_seconds_sum", time.perf_counter() - t0)
            METRICS.inc("kv_router_prefill_seconds_count")

    async def proxy(self, request: web.Request) -> web.StreamResponse:
        path = request.path
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        hashes = prefix_hashes(request_text(body), self.s.block_chars, self.s.max_blocks)

        # Take a decode slot before the prefill hop, so queued requests don't
        # flood the prefill pool either.
        try:
            w = await self.admission.acquire(hashes)
        except QueueFull:
            return web.json_response({"error": "router queue full"}, status=429, headers={"Retry-After": "1"})
        except asyncio.TimeoutError:
            return web.json_response({"error": "timed out waiting in router queue"}, status=503,
                                     headers={"Retry-After": "5"})
        try:
            if self.prefill:
                await self._prefill(path, body, hashes)  # swallows its own errors
            METRICS.inc("kv_router_requests_total", pool="decode", worker=w.url)
            async with self.session.post(f"{w.url}{path}", json=body) as r:
                ctype = r.headers.get("Content-Type", "application/json")
                if not body.get("stream") or "text/event-stream" not in ctype:
                    return web.Response(body=await r.read(), status=r.status,
                                        headers={"Content-Type": ctype})
                resp = web.StreamResponse(status=r.status, headers={
                    "Content-Type": ctype, "Cache-Control": "no-cache", "X-Routed-To": w.url})
                await resp.prepare(request)
                async for chunk in r.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp
        except (ConnectionResetError, asyncio.CancelledError):
            raise  # client went away
        except Exception as e:  # noqa: BLE001
            METRICS.inc("kv_router_errors_total", pool="decode", worker=w.url)
            log.warning("decode %s failed: %s", w.url, e)
            return web.json_response({"error": f"upstream {w.url} failed: {e}"}, status=502)
        finally:
            self.admission.release(w)


def build_app(s: Settings) -> web.Application:
    router = Router(s)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(router.start)
    app.on_cleanup.append(router.stop)
    app.router.add_get("/health", router.health)
    app.router.add_get("/metrics", router.metrics)
    app.router.add_get("/v1/models", router.models)
    app.router.add_get("/workers", router.workers)
    app.router.add_post("/v1/completions", router.proxy)
    app.router.add_post("/v1/chat/completions", router.proxy)
    app["router"] = router
    return app


async def _serve(s: Settings) -> None:
    app = build_app(s)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, s.host, s.port).start()
    if s.metrics_port and s.metrics_port != s.port:
        await web.TCPSite(runner, s.host, s.metrics_port).start()
    log.info("kv_router listening on :%s (metrics :%s) policy=%s decode=%s prefill=%s",
             s.port, s.metrics_port, s.policy, s.decode_urls, s.prefill_urls or "-")
    await asyncio.Event().wait()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_serve(Settings()))
