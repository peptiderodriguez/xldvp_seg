"""HTTP server and Cloudflare tunnel management for the segmentation pipeline.

Handles starting/stopping background HTTP servers, Cloudflare tunnels for remote
access, PID file management, and server status display.
"""

import os
import re
import json
import time
import signal
import atexit
import subprocess
from pathlib import Path

from segmentation.utils.logging import get_logger

logger = get_logger(__name__)

# Global list to track spawned processes for cleanup (foreground mode only)
_spawned_processes = []

# PID file directory for background servers (one file per port)
SERVER_PID_DIR = Path.home() / '.segmentation_servers'
# Legacy single PID file (for backwards compatibility)
SERVER_PID_FILE = Path.home() / '.segmentation_server.pid'


def _get_pid_file(port: int) -> Path:
    """Get PID file path for a specific port."""
    SERVER_PID_DIR.mkdir(exist_ok=True)
    return SERVER_PID_DIR / f'server_{port}.json'


def _get_all_servers() -> list:
    """Get list of all server info dicts."""
    servers = []

    # Check legacy single PID file first
    if SERVER_PID_FILE.exists():
        try:
            data = json.loads(SERVER_PID_FILE.read_text())
            data['_pid_file'] = SERVER_PID_FILE
            servers.append(data)
        except Exception:
            pass

    # Check new per-port PID files
    if SERVER_PID_DIR.exists():
        for pid_file in SERVER_PID_DIR.glob('server_*.json'):
            try:
                data = json.loads(pid_file.read_text())
                data['_pid_file'] = pid_file
                # Skip if already covered by legacy file (same port)
                if not any(s.get('port') == data.get('port') for s in servers):
                    servers.append(data)
            except Exception:
                pass

    return servers


def _cleanup_processes():
    """Cleanup spawned HTTP server and tunnel processes on exit."""
    for proc in _spawned_processes:
        try:
            if proc.poll() is None:  # Still running
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            pass


atexit.register(_cleanup_processes)


def stop_background_server():
    """Stop all running background servers using _get_all_servers()."""
    servers = _get_all_servers()

    if not servers:
        logger.info("No background server running (no PID files found)")
        return False

    stopped = False
    for data in servers:
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        pid_file = data.get('_pid_file')

        for name, pid in [('HTTP server', http_pid), ('Cloudflare tunnel', tunnel_pid)]:
            if pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(f"Stopped {name} (PID {pid})")
                    stopped = True
                except ProcessLookupError:
                    logger.info(f"{name} (PID {pid}) was not running")
                except PermissionError:
                    logger.error(f"Permission denied stopping {name} (PID {pid})")

        # Clean up PID file
        if pid_file and pid_file.exists():
            try:
                pid_file.unlink()
            except OSError:
                pass

    return stopped


def show_server_status():
    """Show status of all running background servers."""
    servers = _get_all_servers()

    if not servers:
        print("No background servers running")
        return False

    # Filter to only running servers
    running_servers = []
    for data in servers:
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        http_running = http_pid and _pid_exists(http_pid)
        tunnel_running = tunnel_pid and _pid_exists(tunnel_pid)

        if http_running or tunnel_running:
            data['_http_running'] = http_running
            data['_tunnel_running'] = tunnel_running
            running_servers.append(data)

    if not running_servers:
        print("No background servers running")
        return False

    print("=" * 70)
    print(f"ACTIVE SEGMENTATION SERVERS ({len(running_servers)} running)")
    print("=" * 70)

    for i, data in enumerate(running_servers):
        http_pid = data.get('http_pid')
        tunnel_pid = data.get('tunnel_pid')
        url = data.get('url')
        port = data.get('port', 8081)
        slide_name = data.get('slide_name', 'unknown')
        cell_type = data.get('cell_type', 'unknown')
        http_running = data.get('_http_running', False)
        tunnel_running = data.get('_tunnel_running', False)

        # Build human-readable name
        if slide_name and slide_name != 'unknown' and cell_type and cell_type != 'unknown':
            serving_name = f"{slide_name} ({cell_type.upper()})"
        elif slide_name and slide_name != 'unknown':
            serving_name = slide_name
        else:
            serving_name = f"Server on port {port}"

        if i > 0:
            print("-" * 70)

        print(f"\n[{i+1}] {serving_name}")
        print(f"    Slide:      {slide_name}")
        print(f"    Cell Type:  {cell_type}")
        print(f"    Port:       {port}")
        print(f"    Status:     HTTP={'OK' if http_running else 'DOWN'}, Tunnel={'OK' if tunnel_running else 'DOWN'}")
        if url and tunnel_running:
            print(f"    PUBLIC:     {url}")
        print(f"    LOCAL:      http://localhost:{port}")

    print("\n" + "=" * 70)
    print(f"To stop all: python run_segmentation.py --stop-server")
    print("=" * 70)
    return True


def _pid_exists(pid):
    """Check if a process with given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def start_server_and_tunnel(html_dir: Path, port: int = 8081, background: bool = False,
                            slide_name: str = None, cell_type: str = None) -> tuple:
    """Start HTTP server and Cloudflare tunnel for viewing results.

    Args:
        html_dir: Path to the HTML directory to serve
        port: Port for HTTP server (default 8081)
        background: If True, detach processes so they survive script exit
        slide_name: Name of the slide being served (for status display)
        cell_type: Type of cells being detected (for status display)

    Returns:
        Tuple of (http_process, tunnel_process, tunnel_url)
    """
    global _spawned_processes

    html_dir = Path(html_dir)
    if not html_dir.exists():
        logger.error(f"HTML directory does not exist: {html_dir}")
        return None, None, None

    # Check for existing server - reuse tunnel if possible
    existing_tunnel_url = None
    existing_tunnel_pid = None
    if background and SERVER_PID_FILE.exists():
        try:
            data = json.loads(SERVER_PID_FILE.read_text())
            old_http_pid = data.get('http_pid')
            old_tunnel_pid = data.get('tunnel_pid')
            old_port = data.get('port', 8081)
            existing_tunnel_url = data.get('url')

            # Check if tunnel is still running
            tunnel_running = old_tunnel_pid and _pid_exists(old_tunnel_pid)

            if tunnel_running and old_port == port:
                # Tunnel is running on same port - keep it, just restart HTTP server
                logger.info(f"Reusing existing Cloudflare tunnel: {existing_tunnel_url}")
                existing_tunnel_pid = old_tunnel_pid

                # Stop only the HTTP server (not the tunnel)
                if old_http_pid and _pid_exists(old_http_pid):
                    try:
                        os.kill(old_http_pid, signal.SIGTERM)
                        logger.info(f"Stopped old HTTP server (PID {old_http_pid})")
                        time.sleep(0.5)  # Give it time to release the port
                    except Exception:
                        pass
            else:
                # Tunnel not running or different port - start fresh
                logger.info("Starting new server and tunnel...")
                stop_background_server()
                existing_tunnel_url = None
        except Exception:
            stop_background_server()

    # Common args for background mode (detach from parent)
    bg_kwargs = {}
    if background:
        bg_kwargs = {
            'start_new_session': True,  # Detach from parent process group
            'stdin': subprocess.DEVNULL,
        }

    # Start HTTP server
    logger.info(f"Starting HTTP server on port {port}...")
    http_proc = subprocess.Popen(
        ['python', '-m', 'http.server', str(port)],
        cwd=str(html_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **bg_kwargs,
    )
    if not background:
        _spawned_processes.append(http_proc)
    time.sleep(1)  # Give server time to start

    if http_proc.poll() is not None:
        logger.error("HTTP server failed to start")
        return None, None, None

    logger.info(f"HTTP server running: http://localhost:{port}")

    # Start Cloudflare tunnel (or reuse existing one)
    tunnel_proc = None
    tunnel_url = existing_tunnel_url

    if existing_tunnel_pid:
        # Reusing existing tunnel - create a dummy process reference
        logger.info(f"Tunnel already running (PID {existing_tunnel_pid})")
        tunnel_proc = None  # We don't have the process object, just the PID

        # Update PID file with new HTTP server but keep tunnel info
        SERVER_PID_FILE.write_text(json.dumps({
            'http_pid': http_proc.pid,
            'tunnel_pid': existing_tunnel_pid,
            'port': port,
            'html_dir': str(html_dir),
            'url': existing_tunnel_url,
            'slide_name': slide_name,
            'cell_type': cell_type,
        }))
    else:
        # Need to start a new tunnel
        cloudflared_path = os.path.expanduser('~/cloudflared')
        if not os.path.exists(cloudflared_path):
            logger.warning("Cloudflare tunnel not found at ~/cloudflared")
            logger.info("Install with: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/cloudflared && chmod +x ~/cloudflared")
            if background:
                # Save HTTP server PID even without tunnel
                SERVER_PID_FILE.write_text(json.dumps({
                    'http_pid': http_proc.pid,
                    'tunnel_pid': None,
                    'port': port,
                    'html_dir': str(html_dir),
                    'url': None,
                    'slide_name': slide_name,
                    'cell_type': cell_type,
                }))
            return http_proc, None, None

        logger.info("Starting Cloudflare tunnel...")

        # For background mode, we need to capture output to get URL but still detach
        if background:
            # Create a log file for tunnel output
            tunnel_log = html_dir / '.tunnel.log'
            tunnel_log_file = open(tunnel_log, 'w')
            try:
                tunnel_proc = subprocess.Popen(
                    [cloudflared_path, 'tunnel', '--url', f'http://localhost:{port}'],
                    stdout=tunnel_log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                )
                # Wait briefly and parse log for URL
                time.sleep(5)
                tunnel_log_file.flush()
            finally:
                tunnel_log_file.close()
            tunnel_url = None
            try:
                log_content = tunnel_log.read_text()
                match = re.search(r'(https://[^\s]+\.trycloudflare\.com)', log_content)
                if match:
                    tunnel_url = match.group(1)
            except Exception:
                pass
        else:
            tunnel_proc = subprocess.Popen(
                [cloudflared_path, 'tunnel', '--url', f'http://localhost:{port}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _spawned_processes.append(tunnel_proc)

            # Wait for tunnel URL (usually appears within 5-10 seconds)
            tunnel_url = None
            start_time = time.time()
            while time.time() - start_time < 30:
                line = tunnel_proc.stdout.readline()
                if not line:
                    if tunnel_proc.poll() is not None:
                        logger.error("Cloudflare tunnel exited unexpectedly")
                        break
                    continue
                # Look for the tunnel URL in the output
                if 'trycloudflare.com' in line:
                    match = re.search(r'(https://[^\s]+\.trycloudflare\.com)', line)
                    if match:
                        tunnel_url = match.group(1)
                        break

        # Save PID file for background mode
        if background:
            SERVER_PID_FILE.write_text(json.dumps({
                'http_pid': http_proc.pid,
                'tunnel_pid': tunnel_proc.pid if tunnel_proc else None,
                'port': port,
                'html_dir': str(html_dir),
                'url': tunnel_url,
                'slide_name': slide_name,
                'cell_type': cell_type,
            }))

    if tunnel_url:
        logger.info("=" * 60)
        logger.info("REMOTE ACCESS AVAILABLE")
        logger.info("=" * 60)
        logger.info(f"Public URL: {tunnel_url}")
        logger.info(f"Local URL:  http://localhost:{port}")
        if background:
            logger.info("")
            logger.info("Server running in BACKGROUND")
            logger.info(f"To stop: python run_segmentation.py --stop-server")
            logger.info(f"PID file: {SERVER_PID_FILE}")
        else:
            logger.info("")
            logger.info("Press Ctrl+C to stop server and tunnel")
        logger.info("=" * 60)
    else:
        logger.warning("Could not get tunnel URL (tunnel may still be starting)")
        logger.info(f"Local URL: http://localhost:{port}")
        if background:
            logger.info(f"Check tunnel log: {html_dir / '.tunnel.log'}")

    return http_proc, tunnel_proc, tunnel_url


def wait_for_server_shutdown(http_proc, tunnel_proc):
    """Wait for user to press Ctrl+C, then cleanup."""
    if http_proc is None:
        return

    try:
        logger.info("Server running. Press Ctrl+C to exit...")
        while True:
            time.sleep(1)
            # Check if processes are still alive
            if http_proc.poll() is not None:
                logger.warning("HTTP server stopped unexpectedly")
                break
            if tunnel_proc and tunnel_proc.poll() is not None:
                logger.warning("Tunnel stopped unexpectedly")
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        _cleanup_processes()
        logger.info("Server stopped")
