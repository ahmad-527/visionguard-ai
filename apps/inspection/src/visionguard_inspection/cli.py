"""Loopback-only CLI; empty registry is not ready, demo is explicitly manufactured."""

import argparse
from pathlib import Path

from visionguard_inspection.registry import (
    Registry,
    load_registry,
    manufactured_registry,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--registry-sha256")
    parser.add_argument("--manufactured-demo", action="store_true")
    args = parser.parse_args(argv)
    if args.manufactured_demo and args.registry:
        parser.error("Manufactured demo and native registry are mutually exclusive")
    if bool(args.registry) != bool(args.registry_sha256):
        parser.error("A registry requires an independently pinned registry SHA-256")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    registry = manufactured_registry() if args.manufactured_demo else Registry()
    if args.registry:
        registry = load_registry(args.registry, args.registry_sha256)
    import uvicorn

    from visionguard_inspection.service import create_app

    uvicorn.run(
        create_app(registry),
        host=args.host,
        port=args.port,
        workers=1,
        proxy_headers=False,
        limit_concurrency=8,
        timeout_keep_alive=5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
