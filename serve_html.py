#!/usr/bin/env python3
"""
Serve any HTML directory with Cloudflare tunnel.

Starts a local HTTP server and creates a Cloudflare tunnel for remote access.
Uses the pipeline's server module (proper cloudflared discovery, PID tracking).

The tunnel URL is always written to a .url file next to the served directory
so it can be retrieved programmatically.

Usage:
    python serve_html.py /path/to/output/html
    python serve_html.py /path/to/output/html --port 8081
    python serve_html.py /path/to/spatial_viewer.html          # serve parent dir
    python serve_html.py /path/to/output --background          # detach
    python serve_html.py --stop                                 # kill all servers
    python serve_html.py --status                               # show running servers
"""

import argparse
import json
import sys
from pathlib import Path

# Well-known location for the most recent server info
_SERVER_INFO_FILE = Path("/tmp/xldvp_seg_serve_info.json")


def main():
    parser = argparse.ArgumentParser(description="Serve HTML with Cloudflare tunnel")
    parser.add_argument("path", nargs="?", help="Directory or HTML file to serve")
    parser.add_argument("--port", type=int, default=8081, help="Port (default: 8081)")
    parser.add_argument("--no-tunnel", action="store_true", help="HTTP only, no tunnel")
    parser.add_argument("--background", action="store_true", help="Detach server processes")
    parser.add_argument("--stop", action="store_true", help="Stop all background servers")
    parser.add_argument("--status", action="store_true", help="Show running servers")
    parser.add_argument(
        "--get-url", action="store_true", help="Print the URL of the most recent server and exit"
    )

    args = parser.parse_args()

    # Quick URL retrieval — no server startup
    if args.get_url:
        if _SERVER_INFO_FILE.exists():
            import os

            info = json.loads(_SERVER_INFO_FILE.read_text())
            # Check if server process is still alive
            pid = info.get("http_pid")
            if pid:
                try:
                    os.kill(pid, 0)  # signal 0 = liveness check
                except (ProcessLookupError, PermissionError):
                    print("Server process is no longer running (stale info).", file=sys.stderr)
                    _SERVER_INFO_FILE.unlink(missing_ok=True)
                    sys.exit(1)
            url = info.get("tunnel_url") or f"http://localhost:{info.get('port', '?')}"
            print(url)
        else:
            print("No active server found.", file=sys.stderr)
            sys.exit(1)
        return

    from segmentation.pipeline.server import (
        show_server_status,
        start_server_and_tunnel,
        stop_background_server,
        wait_for_server_shutdown,
    )

    if args.stop:
        stop_background_server()
        _SERVER_INFO_FILE.unlink(missing_ok=True)
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
        url = f"http://localhost:{args.port}"
        print(f"Serving {directory} at {url}")
        _write_server_info(directory, args.port, url, http_pid=proc.pid)
        print("Press Ctrl+C to stop")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    # Try requested port, then up to 5 more if busy
    max_retries = 5
    actual_port = args.port
    for attempt in range(max_retries + 1):
        try_port = args.port + attempt
        http_proc, tunnel_proc, tunnel_url = start_server_and_tunnel(
            directory,
            port=try_port,
            background=args.background,
            slide_name=slide_name,
            cell_type="serve_html",
        )
        if http_proc is not None:
            actual_port = try_port
            break
        if attempt < max_retries:
            print(f"Port {try_port} busy, trying {try_port + 1}...")
    else:
        print(
            f"ERROR: Could not find a free port ({args.port}-{args.port + max_retries})",
            file=sys.stderr,
        )
        sys.exit(1)

    if actual_port != args.port:
        print(f"Using port {actual_port} (requested {args.port} was busy)")

    # Always print the URL prominently
    if tunnel_url:
        print(f"\n{'=' * 60}")
        print(f"  TUNNEL URL: {tunnel_url}")
        print(f"{'=' * 60}")
        print(f"  Local:  http://localhost:{actual_port}")
        print(f"  Remote: {tunnel_url}")
        print(f"  Dir:    {directory}")
        print(f"{'=' * 60}\n")
    else:
        print(f"\nServing: {directory}")
        print(f"Local only: http://localhost:{actual_port}")
        print("(Cloudflare tunnel not available)\n")

    # Write server info to well-known file for programmatic retrieval
    _write_server_info(directory, actual_port, tunnel_url, http_pid=http_proc.pid)

    if not args.background:
        wait_for_server_shutdown(http_proc, tunnel_proc)


def _write_server_info(directory, port, tunnel_url, http_pid=None):
    """Write server info to a well-known file for programmatic URL retrieval."""
    info = {
        "directory": str(directory),
        "port": port,
        "tunnel_url": tunnel_url,
        "local_url": f"http://localhost:{port}",
        "http_pid": http_pid,
    }
    try:
        _SERVER_INFO_FILE.write_text(json.dumps(info, indent=2))
    except OSError:
        pass  # /tmp write failure is non-fatal


if __name__ == "__main__":
    main()
