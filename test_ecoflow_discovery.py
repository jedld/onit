"""
Quick interactive test for EcoFlow PowerStream auto-discovery and telemetry.
Run: python3 test_ecoflow_discovery.py
"""

import os
import sys
import json
import time

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from src.mcp.servers.tasks.iot.ecoflow.mcp_server import (
    _get_local_subnet,
    _probe_port,
    _probe_ecoflow_mqtt,
    _discover_device,
    _ensure_connected,
    _snapshot,
    get_solar_generation,
    get_power_status,
    get_battery_status,
    get_device_info,
    get_raw_telemetry,
)

SEP = "─" * 60


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def pretty(raw: str):
    try:
        print(json.dumps(json.loads(raw), indent=2))
    except Exception:
        print(raw)


# ── 1. Subnet detection ────────────────────────────────────────────────────
section("1 · Local subnet detection")
subnet = _get_local_subnet()
print(f"Detected subnet: {subnet}")

# ── 2. Port scan ───────────────────────────────────────────────────────────
section("2 · Port-1883 scan on " + (subnet or "?"))
import concurrent.futures, ipaddress
network = ipaddress.IPv4Network(subnet, strict=False)
hosts = [str(h) for h in network.hosts()]
print(f"Scanning {len(hosts)} hosts (timeout 0.4 s each) …")

open_hosts = []
with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
    futures = {ex.submit(_probe_port, ip, 1883, 0.4): ip for ip in hosts}
    for fut in concurrent.futures.as_completed(futures):
        ip = futures[fut]
        try:
            if fut.result():
                open_hosts.append(ip)
                print(f"  ✓ port 1883 open: {ip}")
        except Exception:
            pass

if not open_hosts:
    print("  No hosts found with port 1883 open.")

# ── 3. MQTT identity probe ─────────────────────────────────────────────────
section("3 · EcoFlow MQTT identity probe")
found = None
for ip in open_hosts:
    print(f"  Probing {ip} for EcoFlow topics (up to 4 s) …")
    sn = _probe_ecoflow_mqtt(ip, 1883, timeout=4.0)
    if sn:
        print(f"  ✓ EcoFlow device found!  IP={ip}  SN={sn}")
        found = {"ip": ip, "sn": sn}
        break
    else:
        print(f"    └─ {ip}: no EcoFlow topics detected")

if not found:
    print("  No EcoFlow device identified.")
    sys.exit(1)

# ── 4. Connect and read telemetry ──────────────────────────────────────────
section("4 · Connecting to device")
os.environ["ECOFLOW_DEVICE_IP"] = found["ip"]
os.environ["ECOFLOW_DEVICE_SN"] = found["sn"]
os.environ.setdefault("ECOFLOW_MQTT_USER", found["sn"])

_ensure_connected()
# Wait briefly for first telemetry message
print("Waiting up to 8 s for first telemetry …")
for i in range(16):
    snap = _snapshot()
    if snap["last_seen"]:
        print(f"  ✓ First message received at {snap['last_seen']}")
        break
    time.sleep(0.5)
else:
    print("  ⚠ No telemetry received yet (device may require credentials)")

# ── 5. Tool outputs ────────────────────────────────────────────────────────
section("5 · get_device_info")
pretty(get_device_info())

section("6 · get_solar_generation")
pretty(get_solar_generation())

section("7 · get_power_status")
pretty(get_power_status())

section("8 · get_battery_status")
pretty(get_battery_status())

section("9 · get_raw_telemetry")
pretty(get_raw_telemetry())

print(f"\n{SEP}")
print("  All tests complete.")
print(SEP)
