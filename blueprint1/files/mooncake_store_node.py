#!/usr/bin/env python3
"""Dedicated Mooncake Store node.

Joins the Mooncake cluster and contributes MOONCAKE_STORE_GB of DRAM to the
global KV pool (objects evicted from DRAM spill to MOONCAKE_OFFLOAD_FILE_STORAGE_PATH
on NVMe when the master runs with --enable_offload). vLLM pods (via LMCache) put
and get KV chunks; the master places them in this segment.

Exposes GET /health on :8081 for the kubelet probes.
"""
import http.server
import os
import signal
import sys
import threading

from mooncake.store import MooncakeDistributedStore

GB = 1024 ** 3


def main() -> None:
    pod_ip = os.environ["POD_IP"]
    master = os.environ["MOONCAKE_MASTER_ADDR"]
    metadata = os.environ["MOONCAKE_METADATA_SERVER"]
    segment = int(float(os.environ.get("MOONCAKE_STORE_GB", "32")) * GB)
    local_buf = int(float(os.environ.get("MOONCAKE_LOCAL_BUFFER_GB", "1")) * GB)
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "tcp")

    store = MooncakeDistributedStore()
    rc = store.setup(pod_ip, metadata, segment, local_buf, protocol, "", master)
    if rc != 0:
        sys.exit(f"Mooncake store setup failed (rc={rc}) master={master} metadata={metadata}")
    print(f"mooncake store node ready: host={pod_ip} segment={segment / GB:.0f}GB "
          f"master={master} protocol={protocol}", flush=True)

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()
            self.wfile.write(b"ok\n")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", 8081), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    print("shutting down store node", flush=True)
    store.close()


if __name__ == "__main__":
    main()
