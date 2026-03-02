'''
# Copyright 2025 Joseph Emmanuel Dayo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

EcoFlow PowerStream Local MCP Server

Reads solar generation and power-flow data from an EcoFlow PowerStream device
via its **local** MQTT broker.  No EcoFlow cloud account is required.

--- Auto-discovery ---
If ECOFLOW_DEVICE_IP / ECOFLOW_DEVICE_SN are not set the server will
automatically scan the local /24 subnet for an open port 1883, then probe
each candidate with a brief anonymous MQTT connection to detect the EcoFlow
topic pattern and extract the device serial number.  Discovery runs once at
start-up and caches the result; call discover_device() to re-scan.

--- Local MQTT setup ---
1. Connect to EcoFlow app → device settings → "Local API" (or note the device
   IP from your router DHCP table).
2. The device runs a Mosquitto broker on port 1883.
3. Credentials:
   - Username : device serial number  (e.g. HW51xxxxxxxxxxxxxxxx)
   - Password : local device password (visible in app or obtained via
                the one-time cloud bootstrap described below)

Required environment variables (manual config, all optional if auto-discovery
works):
    ECOFLOW_DEVICE_IP   – LAN IP of the PowerStream  (e.g. 192.168.1.42)
    ECOFLOW_DEVICE_SN   – Serial number              (e.g. HW51xxxxxxxxxxxxxxxx)
    ECOFLOW_MQTT_PASS   – Local MQTT password

Optional:
    ECOFLOW_MQTT_PORT   – defaults to 1883
    ECOFLOW_MQTT_USER   – defaults to ECOFLOW_DEVICE_SN

Requires:
    pip install fastmcp paho-mqtt

6 Core Tools:
1. get_solar_generation  – Live PV1 + PV2 solar input watts
2. get_power_status      – Full power-flow snapshot (solar, battery, grid, load)
3. get_battery_status    – Battery SoC, charge/discharge rate, voltage
4. get_device_info       – Device metadata and connection status
5. get_raw_telemetry     – All decoded fields from the latest MQTT message
6. discover_device          – Scan the LAN and auto-detect the PowerStream
7. get_local_credentials    – One-time bootstrap: fetch MQTT password from EcoFlow cloud API
'''

import concurrent.futures
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import socket
import struct
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    raise ImportError(
        "`paho-mqtt` not installed.  Please run: pip install paho-mqtt"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PowerStream protobuf field → human name + scale (raw → watts / % / V / A)
# Field numbers sourced from community reverse-engineering of EFT210 / EFT400.
# Values arrive as int32 milliwatts (power) or centi-percent (SoC), etc.
_PW_FIELDS: Dict[int, Dict[str, Any]] = {
    # Solar input
    1:  {"name": "pv1_input_watts",      "scale": 0.1,  "unit": "W"},
    2:  {"name": "pv2_input_watts",      "scale": 0.1,  "unit": "W"},
    3:  {"name": "pv1_input_voltage",    "scale": 0.1,  "unit": "V"},
    4:  {"name": "pv2_input_voltage",    "scale": 0.1,  "unit": "V"},
    5:  {"name": "pv1_input_current",    "scale": 0.01, "unit": "A"},
    6:  {"name": "pv2_input_current",    "scale": 0.01, "unit": "A"},
    # Battery
    7:  {"name": "bat_soc",              "scale": 1.0,  "unit": "%"},
    8:  {"name": "bat_input_watts",      "scale": 0.1,  "unit": "W"},
    9:  {"name": "bat_output_watts",     "scale": 0.1,  "unit": "W"},
    10: {"name": "bat_voltage",          "scale": 0.1,  "unit": "V"},
    11: {"name": "bat_current",          "scale": 0.01, "unit": "A"},
    12: {"name": "bat_temperature",      "scale": 0.1,  "unit": "°C"},
    # Inverter / grid / load
    13: {"name": "inv_output_watts",     "scale": 0.1,  "unit": "W"},
    14: {"name": "grid_cons_watts",      "scale": 0.1,  "unit": "W"},
    15: {"name": "pv_to_inv_watts",      "scale": 0.1,  "unit": "W"},
    16: {"name": "load_watts",           "scale": 0.1,  "unit": "W"},
    17: {"name": "inv_frequency",        "scale": 0.1,  "unit": "Hz"},
    18: {"name": "inv_voltage",          "scale": 0.1,  "unit": "V"},
    # Misc
    19: {"name": "heat_sink_temperature","scale": 0.1,  "unit": "°C"},
    20: {"name": "rated_power",          "scale": 1.0,  "unit": "W"},
    21: {"name": "pv_input_total_watts", "scale": 0.1,  "unit": "W"},
    22: {"name": "permanent_watts",      "scale": 0.1,  "unit": "W"},
    23: {"name": "dynamic_watts",        "scale": 0.1,  "unit": "W"},
}

# EcoFlow serial-number patterns found in MQTT topics
_SN_PATTERNS: List[str] = [
    r"/app/[^/]+/thing/(HW[0-9A-Z]{8,})/",
    r"/(HW[0-9A-Z]{8,})/app/",
    r"/app/device/property/(HW[0-9A-Z]{8,})",
    r"(HW[0-9A-Z]{10,})",  # fallback: any HW-prefixed token in the topic
]

# ---------------------------------------------------------------------------
# Network / auto-discovery helpers
# ---------------------------------------------------------------------------

def _get_local_subnet() -> Optional[str]:
    """Return the local network in CIDR notation (e.g. '192.168.1.0/24')."""
    # Preferred: read default-route interface from /proc/net/route (Linux)
    try:
        import subprocess
        with open("/proc/net/route") as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split()
                # Destination == 00000000 is the default route
                if len(parts) >= 8 and parts[1] == "00000000":
                    iface = parts[0]
                    result = subprocess.run(
                        ["ip", "-4", "addr", "show", iface],
                        capture_output=True, text=True, timeout=3,
                    )
                    for l in result.stdout.splitlines():
                        l = l.strip()
                        if l.startswith("inet "):
                            cidr = l.split()[1]  # e.g. "192.168.1.100/24"
                            return str(ipaddress.IPv4Network(cidr, strict=False))
    except Exception:
        pass
    # Fallback: UDP trick — no packet is sent but reveals outbound interface IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return str(ipaddress.IPv4Network(f"{local_ip}/24", strict=False))
    except Exception:
        return None


def _probe_port(ip: str, port: int = 1883, timeout: float = 0.4) -> bool:
    """Return True if *port* is open on *ip*."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except Exception:
        return False


def _probe_ecoflow_mqtt(ip: str, port: int = 1883,
                        timeout: float = 4.0) -> Optional[str]:
    """
    Open an anonymous MQTT connection to *ip*:*port*, subscribe to '#',
    and wait up to *timeout* seconds for an EcoFlow-pattern topic.
    Returns the device serial number string on success, else None.
    """
    found_sn: List[Optional[str]] = [None]
    done = threading.Event()

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe("#", qos=0)

    def on_message(client, userdata, msg):
        for pat in _SN_PATTERNS:
            m = re.search(pat, msg.topic)
            if m:
                found_sn[0] = m.group(1)
                done.set()
                return

    probe = mqtt.Client(
        client_id="onit-ecoflow-probe",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        protocol=mqtt.MQTTv311,
    )
    probe.on_connect = on_connect
    probe.on_message = on_message
    try:
        probe.connect(ip, port, keepalive=10)
        probe.loop_start()
        done.wait(timeout=timeout)
    except Exception:
        pass
    finally:
        try:
            probe.loop_stop()
            probe.disconnect()
        except Exception:
            pass
    return found_sn[0]


def _discover_device(
    subnet: Optional[str] = None,
    mqtt_port: int = 1883,
    scan_timeout: float = 0.4,
    probe_timeout: float = 4.0,
    max_workers: int = 64,
) -> Optional[Dict[str, str]]:
    """
    Scan *subnet* (/24 by default = local network) for open port *mqtt_port*,
    then probe each candidate for EcoFlow topic patterns.
    Returns {"ip": ..., "sn": ...} on success, else None.
    """
    if subnet is None:
        subnet = _get_local_subnet()
    if not subnet:
        logger.error("Could not determine local subnet for discovery")
        return None

    network = ipaddress.IPv4Network(subnet, strict=False)
    # Skip /8 or larger to avoid accidental wide scans
    if network.prefixlen < 16:
        logger.error("Subnet %s too broad for auto-discovery (min /16)", subnet)
        return None

    hosts = [str(h) for h in network.hosts()]
    logger.info("EcoFlow discovery: scanning %d hosts on %s for port %d",
                len(hosts), subnet, mqtt_port)

    open_hosts: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_probe_port, ip, mqtt_port, scan_timeout): ip
            for ip in hosts
        }
        for fut in concurrent.futures.as_completed(futures):
            ip = futures[fut]
            try:
                if fut.result():
                    open_hosts.append(ip)
                    logger.info("EcoFlow discovery: port %d open on %s",
                                mqtt_port, ip)
            except Exception:
                pass

    if not open_hosts:
        logger.info("EcoFlow discovery: no hosts found with port %d open", mqtt_port)
        return None

    # Probe each open host — stop at first EcoFlow device found
    for ip in open_hosts:
        logger.info("EcoFlow discovery: probing %s for EcoFlow topics", ip)
        sn = _probe_ecoflow_mqtt(ip, mqtt_port, probe_timeout)
        if sn:
            logger.info("EcoFlow discovery: found device SN=%s at %s", sn, ip)
            return {"ip": ip, "sn": sn}

    logger.info("EcoFlow discovery: no EcoFlow device identified on scanned hosts")
    return None


# ---------------------------------------------------------------------------
# State shared between MQTT background thread and MCP tools
# ---------------------------------------------------------------------------

_state: Dict[str, Any] = {
    "connected": False,
    "last_seen": None,
    "topic": None,
    "fields": {},       # decoded field_name → {"value": float, "unit": str}
    "raw_json": None,   # if message was JSON
    "error": None,
}
_state_lock = threading.Lock()
_mqtt_client: Optional[mqtt.Client] = None

# ---------------------------------------------------------------------------
# Protobuf helpers (minimal varint / wire-type decoder, no external deps)
# ---------------------------------------------------------------------------

def _read_varint(data: bytes, pos: int):
    """Decode a base-128 varint from *data* starting at *pos*.
    Returns (value, new_pos) or raises ValueError on truncation."""
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("Varint too long")
    raise ValueError("Truncated varint")


def _decode_protobuf_flat(data: bytes) -> Dict[int, int]:
    """
    Shallow protobuf decode: extract all (varint / 32-bit / 64-bit) leaf
    fields from *data* and return {field_number: int_value}.
    Skips nested length-delimited fields gracefully.
    """
    fields: Dict[int, int] = {}
    pos = 0
    while pos < len(data):
        try:
            tag, pos = _read_varint(data, pos)
        except ValueError:
            break
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:          # varint
            val, pos = _read_varint(data, pos)
            fields[field_num] = val
        elif wire_type == 1:        # 64-bit
            if pos + 8 > len(data):
                break
            val = struct.unpack_from("<Q", data, pos)[0]
            fields[field_num] = val
            pos += 8
        elif wire_type == 2:        # length-delimited (skip or recurse)
            length, pos = _read_varint(data, pos)
            if pos + length > len(data):
                break
            # Try to recurse into nested message; ignore failures
            try:
                nested = _decode_protobuf_flat(data[pos:pos + length])
                for k, v in nested.items():
                    fields.setdefault(k, v)
            except Exception:
                pass
            pos += length
        elif wire_type == 5:        # 32-bit
            if pos + 4 > len(data):
                break
            val = struct.unpack_from("<I", data, pos)[0]
            fields[field_num] = val
            pos += 4
        else:
            break   # unknown / corrupt wire type; stop
    return fields


def _apply_field_map(raw_fields: Dict[int, int]) -> Dict[str, Any]:
    """Convert raw protobuf integer values to named, scaled readings."""
    result: Dict[str, Any] = {}
    for fnum, info in _PW_FIELDS.items():
        if fnum in raw_fields:
            scaled = round(raw_fields[fnum] * info["scale"], 2)
            result[info["name"]] = {"value": scaled, "unit": info["unit"]}
    return result


# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------

def _on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        with _state_lock:
            _state["connected"] = True
            _state["error"] = None
        sn = userdata.get("sn", "#")
        # Subscribe to all topics from this device
        for topic in [f"#", f"/app/+/thing/{sn}/property",
                      f"/{sn}/app/get_message",
                      f"/app/device/property/{sn}"]:
            try:
                client.subscribe(topic, qos=0)
            except Exception:
                pass
        logger.info("EcoFlow MQTT connected, subscribed to device topics")
    else:
        with _state_lock:
            _state["connected"] = False
            _state["error"] = f"MQTT connect failed (rc={rc})"
        logger.error("MQTT connect failed rc=%s", rc)


def _on_disconnect(client, userdata, rc, properties=None):
    with _state_lock:
        _state["connected"] = False
        if rc != 0:
            _state["error"] = f"Unexpected disconnect (rc={rc})"


def _on_message(client, userdata, msg):
    payload = msg.payload
    now = datetime.now().isoformat(timespec="seconds")

    # Try JSON first (some firmware versions broadcast JSON directly)
    decoded_json = None
    try:
        decoded_json = json.loads(payload.decode("utf-8", errors="replace"))
    except Exception:
        pass

    if decoded_json and isinstance(decoded_json, dict):
        # Flatten nested dicts into field name → value
        flat: Dict[str, Any] = {}
        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten(v, key + ".")
                else:
                    flat[key] = v
        _flatten(decoded_json)
        with _state_lock:
            _state["last_seen"] = now
            _state["topic"] = msg.topic
            _state["raw_json"] = decoded_json
            # Map known JSON key names to our fields dict
            _state["fields"].update({
                k: {"value": v, "unit": ""} for k, v in flat.items()
                if isinstance(v, (int, float))
            })
        return

    # Protobuf decode
    try:
        raw = _decode_protobuf_flat(payload)
        mapped = _apply_field_map(raw)
        with _state_lock:
            _state["last_seen"] = now
            _state["topic"] = msg.topic
            _state["raw_json"] = None
            _state["fields"].update(mapped)
    except Exception as exc:
        logger.debug("Could not decode MQTT payload: %s", exc)


# ---------------------------------------------------------------------------
# MQTT connection management
# ---------------------------------------------------------------------------

def _get_config() -> Dict[str, str]:
    ip   = os.environ.get("ECOFLOW_DEVICE_IP", "")
    sn   = os.environ.get("ECOFLOW_DEVICE_SN", "")
    pwd  = os.environ.get("ECOFLOW_MQTT_PASS", "")
    user = os.environ.get("ECOFLOW_MQTT_USER", sn)
    port = int(os.environ.get("ECOFLOW_MQTT_PORT", "1883"))
    return {"ip": ip, "sn": sn, "password": pwd, "user": user, "port": port}


def _load_cached_credentials():
    """Load persisted credentials from ~/.onit/ecoflow_creds.json if present."""
    if not os.path.exists(_CREDS_CACHE_PATH):
        return
    try:
        with open(_CREDS_CACHE_PATH) as f:
            data = json.load(f)
        for env, key in [
            ("ECOFLOW_MQTT_USER", "mqtt_user"),
            ("ECOFLOW_MQTT_PASS", "mqtt_pass"),
        ]:
            if key in data and not os.environ.get(env):
                os.environ[env] = data[key]
        logger.info("Loaded cached EcoFlow credentials from %s", _CREDS_CACHE_PATH)
    except Exception as exc:
        logger.warning("Could not load cached credentials: %s", exc)


def _ensure_connected():
    """Start the MQTT background thread if not already running.

    If ECOFLOW_DEVICE_IP or ECOFLOW_DEVICE_SN are absent, automatically
    scans the local network to find the device.  Cached credentials are
    loaded from ~/.onit/ecoflow_creds.json if present.
    """
    global _mqtt_client
    with _state_lock:
        if _state["connected"]:
            return
    _load_cached_credentials()
    cfg = _get_config()
    if not cfg["ip"] or not cfg["sn"]:
        with _state_lock:
            _state["error"] = "Auto-discovery in progress…"
        logger.info("ECOFLOW_DEVICE_IP/SN not set — running auto-discovery")
        found = _discover_device(
            mqtt_port=cfg["port"] if cfg["port"] else 1883
        )
        if found:
            os.environ["ECOFLOW_DEVICE_IP"] = found["ip"]
            os.environ["ECOFLOW_DEVICE_SN"] = found["sn"]
            os.environ.setdefault("ECOFLOW_MQTT_USER", found["sn"])
            cfg = _get_config()  # re-read with discovered values
            with _state_lock:
                _state["error"] = None
        else:
            with _state_lock:
                _state["error"] = (
                    "Auto-discovery found no EcoFlow device on the local network. "
                    "Set ECOFLOW_DEVICE_IP and ECOFLOW_DEVICE_SN manually."
                )
            return
    try:
        if _mqtt_client:
            try:
                _mqtt_client.disconnect()
            except Exception:
                pass
        client = mqtt.Client(
            client_id=f"onit-mcp-{cfg['sn'][:8]}",
            userdata={"sn": cfg["sn"]},
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311,
        )
        if cfg["user"] and cfg["password"]:
            client.username_pw_set(cfg["user"], cfg["password"])
        client.on_connect    = _on_connect
        client.on_disconnect = _on_disconnect
        client.on_message    = _on_message
        client.connect_async(cfg["ip"], cfg["port"], keepalive=60)
        client.loop_start()
        _mqtt_client = client
        # Give it up to 5 s to connect
        for _ in range(50):
            time.sleep(0.1)
            with _state_lock:
                if _state["connected"]:
                    break
    except Exception as exc:
        with _state_lock:
            _state["error"] = str(exc)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("EcoFlow PowerStream Local MCP Server")


def _snapshot() -> Dict[str, Any]:
    """Return a copy of current state (thread-safe)."""
    with _state_lock:
        return {
            "connected": _state["connected"],
            "last_seen":  _state["last_seen"],
            "topic":      _state["topic"],
            "fields":     dict(_state["fields"]),
            "error":      _state["error"],
        }


def _field(fields: Dict, *names) -> Optional[float]:
    """Return first matching field value, or None."""
    for n in names:
        if n in fields:
            return fields[n]["value"]
    return None


# ---------------------------------------------------------------------------
# EcoFlow cloud credential bootstrap
# ---------------------------------------------------------------------------

_ECOFLOW_API_BASE = "https://api-e.ecoflow.com"
_CREDS_CACHE_PATH = os.path.expanduser("~/.onit/ecoflow_creds.json")


def _ecoflow_sign(access_key: str, secret_key: str,
                  params: Dict[str, str]) -> Dict[str, str]:
    """
    Build the HMAC-SHA256 signed headers required by EcoFlow OpenAPI.
    Signature = HMAC-SHA256(sorted_query_string, secret_key).
    """
    ts = str(int(time.time() * 1000))
    nonce = hashlib.md5(ts.encode()).hexdigest()[:8]
    sign_params = dict(params)
    sign_params["accessKey"] = access_key
    sign_params["timestamp"] = ts
    sign_params["nonce"] = nonce
    # Build canonical query string: sorted keys joined by &
    query = "&".join(f"{k}={sign_params[k]}" for k in sorted(sign_params))
    sig = hmac.new(secret_key.encode(), query.encode(), hashlib.sha256).hexdigest()
    return {
        "accessKey": access_key,
        "timestamp": ts,
        "nonce": nonce,
        "sign": sig,
    }


def _fetch_cloud_credentials(access_key: str,
                              secret_key: str) -> Dict[str, Any]:
    """
    Call EcoFlow OpenAPI /iot-open/sign/certification to retrieve the
    MQTT credentials that also authenticate to the device's local broker.

    Returns dict with keys: url, port, certificateAccount, certificatePassword
    """
    headers = _ecoflow_sign(access_key, secret_key, {})
    qs = urllib.parse.urlencode(headers)
    url = f"{_ECOFLOW_API_BASE}/iot-open/sign/certification?{qs}"
    req = urllib.request.Request(
        url,
        headers={
            "Content-Type": "application/json",
            "accessKey": headers["accessKey"],
            "timestamp": headers["timestamp"],
            "nonce":     headers["nonce"],
            "sign":      headers["sign"],
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
    if body.get("code") != "0" and body.get("code") != 0:
        raise ValueError(f"EcoFlow API error: {body.get('message', body)}")
    return body["data"]


@mcp.tool(
    title="Get Local Credentials",
    description="""One-time bootstrap: fetch the local MQTT password from EcoFlow's cloud API.

The EcoFlow PowerStream's local MQTT broker requires credentials that are
provisioned by EcoFlow's servers — they cannot be guessed or derived from
the serial number.  This tool calls the EcoFlow OpenAPI to retrieve them.

Requires a free EcoFlow developer account:
  https://developer.ecoflow.com  →  'Create App'  →  copy Access Key + Secret Key

Args:
- access_key: EcoFlow developer Access Key
- secret_key: EcoFlow developer Secret Key

On success the credentials are:
- Saved to ~/.onit/ecoflow_creds.json
- Set in the current process environment (ECOFLOW_MQTT_USER / ECOFLOW_MQTT_PASS)
- Used immediately to connect to the device at ECOFLOW_DEVICE_IP

Returns JSON: {mqtt_user, mqtt_host, mqtt_port, device_ip, status}"""
)
def get_local_credentials(access_key: str = "", secret_key: str = "") -> str:
    if not access_key or not secret_key:
        return json.dumps({
            "status": "error",
            "error": (
                "access_key and secret_key are required. "
                "Get them free at https://developer.ecoflow.com"
            ),
        })
    try:
        data = _fetch_cloud_credentials(access_key, secret_key)
    except Exception as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    mqtt_user = data.get("certificateAccount", "")
    mqtt_pass = data.get("certificatePassword", "")
    mqtt_host = data.get("url", "mqtt.ecoflow.com")
    mqtt_port = str(data.get("port", 8883))

    # Persist to disk
    os.makedirs(os.path.dirname(_CREDS_CACHE_PATH), exist_ok=True)
    with open(_CREDS_CACHE_PATH, "w") as f:
        json.dump({
            "mqtt_user": mqtt_user,
            "mqtt_pass": mqtt_pass,
            "mqtt_host": mqtt_host,
            "mqtt_port": mqtt_port,
        }, f, indent=2)
    os.chmod(_CREDS_CACHE_PATH, 0o600)

    # Inject into environment for this session
    os.environ["ECOFLOW_MQTT_USER"] = mqtt_user
    os.environ["ECOFLOW_MQTT_PASS"] = mqtt_pass
    # Use the local device IP if already known, otherwise fall back to cloud host
    device_ip = os.environ.get("ECOFLOW_DEVICE_IP", "")
    if not device_ip:
        os.environ["ECOFLOW_DEVICE_IP"] = mqtt_host

    # Reconnect with new credentials
    _ensure_connected()

    return json.dumps({
        "status":    "ok",
        "mqtt_user": mqtt_user,
        "mqtt_host": mqtt_host,
        "mqtt_port": int(mqtt_port),
        "device_ip": os.environ.get("ECOFLOW_DEVICE_IP"),
        "saved_to":  _CREDS_CACHE_PATH,
    })


@mcp.tool(
    title="Get Solar Generation",
    description="""Return current solar generation data from the EcoFlow PowerStream.

Reports:
- PV1 and PV2 individual channel watts
- Combined total solar input watts
- Last update timestamp

Returns JSON: {pv1_watts, pv2_watts, total_solar_watts, timestamp, status}"""
)
def get_solar_generation() -> str:
    _ensure_connected()
    snap = _snapshot()
    f = snap["fields"]

    pv1   = _field(f, "pv1_input_watts")
    pv2   = _field(f, "pv2_input_watts")
    total = _field(f, "pv_input_total_watts")
    if total is None and pv1 is not None and pv2 is not None:
        total = round(pv1 + pv2, 2)

    if not snap["connected"] and not snap["last_seen"]:
        return json.dumps({
            "status": "error",
            "error": snap["error"] or "Not connected to device",
        })

    return json.dumps({
        "pv1_watts":         pv1,
        "pv2_watts":         pv2,
        "total_solar_watts": total,
        "timestamp":         snap["last_seen"],
        "connected":         snap["connected"],
        "status":            "ok" if snap["last_seen"] else "waiting_for_data",
    })


@mcp.tool(
    title="Get Power Status",
    description="""Return a full power-flow snapshot from the EcoFlow PowerStream.

Reports:
- Solar input (PV1, PV2, total)
- Battery: SoC %, charge/discharge watts
- Inverter output watts
- Grid consumption watts
- Estimated home load watts
- Last update timestamp

Returns JSON with all available power metrics."""
)
def get_power_status() -> str:
    _ensure_connected()
    snap = _snapshot()
    f = snap["fields"]

    if not snap["connected"] and not snap["last_seen"]:
        return json.dumps({
            "status": "error",
            "error": snap["error"] or "Not connected to device",
        })

    pv1   = _field(f, "pv1_input_watts")
    pv2   = _field(f, "pv2_input_watts")
    total = _field(f, "pv_input_total_watts")
    if total is None and pv1 is not None and pv2 is not None:
        total = round(pv1 + pv2, 2)

    bat_in  = _field(f, "bat_input_watts")
    bat_out = _field(f, "bat_output_watts")
    bat_net = None
    if bat_in is not None and bat_out is not None:
        bat_net = round(bat_in - bat_out, 2)  # positive = charging

    return json.dumps({
        "solar": {
            "pv1_watts":   pv1,
            "pv2_watts":   pv2,
            "total_watts": total,
        },
        "battery": {
            "soc_pct":        _field(f, "bat_soc"),
            "charge_watts":   bat_in,
            "discharge_watts": bat_out,
            "net_watts":      bat_net,
            "voltage_v":      _field(f, "bat_voltage"),
        },
        "inverter": {
            "output_watts":  _field(f, "inv_output_watts"),
            "voltage_v":     _field(f, "inv_voltage"),
            "frequency_hz":  _field(f, "inv_frequency"),
        },
        "grid_consumption_watts": _field(f, "grid_cons_watts"),
        "load_watts":             _field(f, "load_watts"),
        "permanent_watts":        _field(f, "permanent_watts"),
        "dynamic_watts":          _field(f, "dynamic_watts"),
        "timestamp":              snap["last_seen"],
        "connected":              snap["connected"],
        "status":                 "ok" if snap["last_seen"] else "waiting_for_data",
    })


@mcp.tool(
    title="Get Battery Status",
    description="""Return battery-specific data from the EcoFlow PowerStream.

Reports:
- State of charge (%)
- Charge and discharge watts
- Voltage and current
- Temperature

Returns JSON: {soc_pct, charge_watts, discharge_watts, voltage_v, current_a, temperature_c, timestamp, status}"""
)
def get_battery_status() -> str:
    _ensure_connected()
    snap = _snapshot()
    f = snap["fields"]

    if not snap["connected"] and not snap["last_seen"]:
        return json.dumps({
            "status": "error",
            "error": snap["error"] or "Not connected to device",
        })

    return json.dumps({
        "soc_pct":         _field(f, "bat_soc"),
        "charge_watts":    _field(f, "bat_input_watts"),
        "discharge_watts": _field(f, "bat_output_watts"),
        "voltage_v":       _field(f, "bat_voltage"),
        "current_a":       _field(f, "bat_current"),
        "temperature_c":   _field(f, "bat_temperature"),
        "timestamp":       snap["last_seen"],
        "connected":       snap["connected"],
        "status":          "ok" if snap["last_seen"] else "waiting_for_data",
    })


@mcp.tool(
    title="Get Device Info",
    description="""Return EcoFlow PowerStream device metadata and connection status.

Reports:
- Connection status to local MQTT broker
- Device serial number and IP
- Last telemetry timestamp
- Active MQTT topic
- Any connection error messages

Returns JSON: {connected, device_ip, device_sn, mqtt_port, last_seen, topic, error, status}"""
)
def get_device_info() -> str:
    _ensure_connected()
    snap = _snapshot()
    cfg  = _get_config()

    return json.dumps({
        "connected":   snap["connected"],
        "device_ip":   cfg["ip"],
        "device_sn":   cfg["sn"],
        "mqtt_port":   cfg["port"],
        "last_seen":   snap["last_seen"],
        "topic":       snap["topic"],
        "error":       snap["error"],
        "status":      "ok" if snap["connected"] else "disconnected",
    })


@mcp.tool(
    title="Discover Device",
    description=
"""Scan the local network for an EcoFlow PowerStream and return its IP and serial number.

Performs two steps:
1. Concurrent TCP port scan on port 1883 across the local /24 subnet
2. Brief anonymous MQTT probe on each open host to detect EcoFlow topic patterns

Args:
- subnet: CIDR subnet to scan (e.g. "192.168.1.0/24"). Auto-detected if omitted.
- mqtt_port: MQTT port to scan (default: 1883)

Returns JSON: {ip, sn, subnet_scanned, candidates, status}"""
)
def discover_device(subnet: str = "", mqtt_port: int = 1883) -> str:
    result = _discover_device(
        subnet=subnet or None,
        mqtt_port=mqtt_port,
    )
    if result:
        # Persist for this session
        os.environ["ECOFLOW_DEVICE_IP"] = result["ip"]
        os.environ["ECOFLOW_DEVICE_SN"] = result["sn"]
        os.environ.setdefault("ECOFLOW_MQTT_USER", result["sn"])
        # Re-connect with discovered credentials
        _ensure_connected()
        return json.dumps({
            "ip":     result["ip"],
            "sn":     result["sn"],
            "status": "found",
        })
    return json.dumps({
        "status": "not_found",
        "error":  "No EcoFlow device detected on the local network.",
    })


@mcp.tool(
    title="Get Raw Telemetry",
    description="""Return all decoded telemetry fields from the latest MQTT message.

Useful for discovering what data the device is broadcasting, debugging, or
accessing fields not exposed by the other tools (e.g. auxiliary sensors).

Returns JSON: {fields: {name: {value, unit}}, topic, timestamp, status}"""
)
def get_raw_telemetry() -> str:
    _ensure_connected()
    snap = _snapshot()

    if not snap["connected"] and not snap["last_seen"]:
        return json.dumps({
            "status": "error",
            "error": snap["error"] or "Not connected to device",
        })

    return json.dumps({
        "fields":    snap["fields"],
        "topic":     snap["topic"],
        "timestamp": snap["last_seen"],
        "connected": snap["connected"],
        "status":    "ok" if snap["last_seen"] else "waiting_for_data",
    })


# ---------------------------------------------------------------------------
# Entry point (called by run.py)
# ---------------------------------------------------------------------------

def run(transport: str = "sse",
        host: str = "0.0.0.0",
        port: int = 18215,
        path: str = "/sse",
        options: dict = {}):
    """Start the EcoFlow MCP server.  Called by the onit server runner."""
    if options.get("verbose"):
        logging.getLogger().setLevel(logging.DEBUG)

    # Allow options to override env vars so docker-compose / config can set them
    for env_key, opt_key in [
        ("ECOFLOW_DEVICE_IP",  "device_ip"),
        ("ECOFLOW_DEVICE_SN",  "device_sn"),
        ("ECOFLOW_MQTT_PASS",  "mqtt_password"),
        ("ECOFLOW_MQTT_USER",  "mqtt_user"),
        ("ECOFLOW_MQTT_PORT",  "mqtt_port"),
    ]:
        if opt_key in options:
            os.environ.setdefault(env_key, str(options[opt_key]))

    # Eagerly connect so first tool call is fast
    _ensure_connected()

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=host, port=port, path=path)


if __name__ == "__main__":
    run()
