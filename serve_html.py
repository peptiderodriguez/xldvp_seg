#!/usr/bin/env python3
"""
Serve any HTML directory with Cloudflare tunnel.

Starts a local HTTP server and creates a Cloudflare tunnel for remote access.
Uses the pipeline's server module (proper cloudflared discovery, PID tracking).

Usage:
    python serve_html.py /path/to/output/html
    python serve_html.py /path/to/output/html --port 8081
    python serve_html.py /path/to/spatial_viewer.html          # serve parent dir
    python serve_html.py /path/to/output --background          # detach
    python serve_html.py --stop                                 # kill all servers
    python serve_html.py --status                               # show running servers
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Serve HTML with Cloudflare tunnel")
    parser.add_argument("path", nargs="?", help="Directory or HTML file to serve")
    parser.add_argument("--port", type=int, default=8081, help="Port (default: 8081)")
    parser.add_argument("--no-tunnel", action="store_true", help="HTTP only, no tunnel")
    parser.add_argument("--background", action="store_true", help="Detach server processes")
    parser.add_argument("--stop", action="store_true", help="Stop all background servers")
    parser.add_argument("--status", action="store_true", help="Show running servers")

    args = parser.parse_args()

    from segmentation.pipeline.server import (
        show_server_status,
        start_server_and_tunnel,
        stop_background_server,
        wait_for_server_shutdown,
    )

    if args.stop:
        stop_background_server()
        return

    if args.status:
        show_server_status()
        return

    if not args.path:
        parser.error("path is required (unless using --stop or --status)")

    path = Path(args.path).resolve()

    # Accept an HTML file — serve its parent directory
    if path.is_file():
        directory = path.parent
    elif path.is_dir():
        directory = path
        # Check for html/ subdir
        if (directory / "html").is_dir() and not (directory / "spatial_viewer.html").exists():
            directory = directory / "html"
    else:
        print(f"ERROR: Not found: {path}", file=sys.stderr)
        sys.exit(1)

    slide_name = directory.name

    if args.no_tunnel:
        # HTTP only — just start and block
        import subprocess

        proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(args.port)],
            cwd=str(directory),
        )
        print(f"Serving {directory} at http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    print(f"Serving: {directory}")
    print(f"Port: {args.port}")

    http_proc, tunnel_proc, tunnel_url = start_server_and_tunnel(
        directory,
        port=args.port,
        background=args.background,
        slide_name=slide_name,
        cell_type="serve_html",
    )
    if http_proc is None:
        print("ERROR: Failed to start server", file=sys.stderr)
        sys.exit(1)

    if not args.background:
        wait_for_server_shutdown(http_proc, tunnel_proc)


if __name__ == "__main__":
    main()
