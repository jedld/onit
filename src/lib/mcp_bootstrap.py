"""Helpers for starting and probing local MCP servers."""

import os
import socket
import sys
import threading
import time


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False


def mcp_servers_ready(config_data: dict, timeout: float = 15.0) -> bool:
    """Wait for all enabled MCP servers in the agent config to be reachable."""
    from urllib.parse import urlparse

    servers = config_data.get('mcp', {}).get('servers', [])
    endpoints = []
    for server in servers:
        if server.get('enabled', True) and server.get('url'):
            parsed = urlparse(server['url'])
            host = parsed.hostname or '127.0.0.1'
            port = parsed.port or 80
            endpoints.append((host, port))

    if not endpoints:
        return True

    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if all(is_port_open(host, port) for host, port in endpoints):
            return True
        time.sleep(0.5)
    return False


def _start_mcp_servers_background(log_level: str = 'ERROR') -> None:
    """Start configured MCP servers in a background thread."""
    from ..mcp.servers.run import run_servers

    try:
        run_servers(log_level=log_level)
    except Exception:
        pass


def ensure_mcp_servers(config_data: dict, log_level: str = 'ERROR') -> None:
    """Start local MCP servers if needed, then wait briefly for readiness."""
    from urllib.parse import urlparse

    docs_path = config_data.get('documents_path', '')
    if docs_path:
        os.environ['ONIT_DOCUMENTS_PATH'] = docs_path

    servers = config_data.get('mcp', {}).get('servers', [])
    already_running = True
    for server in servers:
        if server.get('enabled', True) and server.get('url'):
            parsed = urlparse(server['url'])
            host = parsed.hostname or '127.0.0.1'
            port = parsed.port or 80
            if not is_port_open(host, port, timeout=0.3):
                already_running = False
                break

    if already_running and servers:
        return

    mcp_thread = threading.Thread(
        target=_start_mcp_servers_background,
        args=(log_level,),
        daemon=True,
    )
    mcp_thread.start()

    if not mcp_servers_ready(config_data, timeout=15.0):
        print("Warning: some MCP servers may not have started in time.", file=sys.stderr)