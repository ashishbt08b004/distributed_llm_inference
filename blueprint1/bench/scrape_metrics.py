#!/usr/bin/env python3
"""Snapshot Prometheus /metrics from several targets into one text file.

Each target's output is preceded by a `# TARGET <url>` line so analyze.py can
compute per-pod deltas between a "before" and an "after" snapshot.
"""
import argparse
import sys
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("targets", nargs="+", help="base URLs, /metrics is appended")
    args = ap.parse_args()
    with open(args.out, "w") as f:
        for base in args.targets:
            url = base.rstrip("/") + "/metrics"
            f.write(f"# TARGET {base}\n")
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    f.write(r.read().decode("utf-8", "replace"))
            except Exception as e:  # noqa: BLE001
                f.write(f"# SCRAPE_ERROR {e}\n")
                print(f"warning: could not scrape {url}: {e}", file=sys.stderr)
            f.write("\n")


if __name__ == "__main__":
    main()
