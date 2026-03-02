"""
EcoFlow MQTT diagnostic — dumps all topics and raw payloads seen on 192.168.0.106:1883
to help identify the correct topic/SN pattern for the MCP server.

Run:
  python3 test_ecoflow_diag.py                      # anonymous / common users
  python3 test_ecoflow_diag.py DEVICE_SN PASSWORD   # with credentials
"""
import sys, time, json, re
import paho.mqtt.client as mqtt

HOST = "192.168.0.106"
PORT = 1883
DURATION = 10  # seconds to listen per attempt

user_arg     = sys.argv[1] if len(sys.argv) > 1 else None
password_arg = sys.argv[2] if len(sys.argv) > 2 else None

# Credential sets to try: list of (user, password)
if user_arg and password_arg:
    trials = [(user_arg, password_arg)]
elif user_arg:
    trials = [(user_arg, "")]
else:
    trials = [("", ""), ("admin", ""), ("ecoflow", ""), ("test", "")]

SEP = "─" * 70
seen_topics: dict = {}

RC_MEANINGS = {
    0: "Connected ✓",
    1: "Refused — unacceptable protocol version",
    2: "Refused — identifier rejected",
    3: "Refused — server unavailable",
    4: "Refused — bad username or password ✗",
    5: "Refused — not authorised ✗",
}


def hex_dump(data: bytes, max_bytes: int = 64) -> str:
    chunk = data[:max_bytes]
    return " ".join(f"{b:02x}" for b in chunk) + (
        f" … (+{len(data) - max_bytes} more bytes)" if len(data) > max_bytes else ""
    )


def try_json(data: bytes):
    try:
        return json.dumps(json.loads(data.decode("utf-8", errors="replace")), indent=4)
    except Exception:
        return None


def make_callbacks(seen: dict):
    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"  on_connect rc={rc}: {RC_MEANINGS.get(rc, 'unknown')}")
        if rc == 0:
            client.subscribe("#", qos=0)
            client.subscribe("$SYS/#", qos=0)

    def on_message(client, userdata, msg):
        topic = msg.topic
        payload = msg.payload
        if topic not in seen:
            seen[topic] = 0
            print(f"\n  NEW TOPIC: {topic}")
            j = try_json(payload)
            if j:
                print(f"  Payload (JSON, {len(payload)}B):")
                for line in j.splitlines()[:30]:
                    print(f"    {line}")
            else:
                print(f"  Payload ({len(payload)}B binary): {hex_dump(payload)}")
        seen[topic] += 1

    return on_connect, on_message


print(f"\n{SEP}")
print("  EcoFlow MQTT Diagnostic")
print(f"  Target  : {HOST}:{PORT}")
print(f"  Listen  : {DURATION}s per attempt")
print(SEP)

for user, password in trials:
    print(f"\nTrying  user={user!r}  password={'(set)' if password else '(empty)'}")
    seen_this: dict = {}
    on_conn, on_msg = make_callbacks(seen_this)

    c = mqtt.Client(
        client_id="onit-diag",
        protocol=mqtt.MQTTv311,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    if user or password:
        c.username_pw_set(user, password)
    c.on_connect = on_conn
    c.on_message = on_msg

    try:
        c.connect(HOST, PORT, keepalive=30)
        c.loop_start()
        time.sleep(DURATION)
        c.loop_stop()
        c.disconnect()
    except Exception as e:
        print(f"  Exception: {e}")

    seen_topics.update(seen_this)
    if seen_this:
        break  # got messages — stop trying

print(f"\n{SEP}")
if seen_topics:
    print(f"  Summary — {len(seen_topics)} unique topics seen:")
    for t, count in sorted(seen_topics.items()):
        print(f"    [{count:4d}]  {t}")

    sn_hits = set()
    for t in seen_topics:
        for pat in [
            r"/(HW[0-9A-Za-z]{8,})/",
            r"/(EFT[0-9A-Za-z]{6,})/",
            r"/thing/([A-Z0-9]{10,})/",
            r"/property/([A-Z0-9]{10,})",
            r"/([A-Z]{2}[0-9A-Z]{8,})/",
        ]:
            m = re.search(pat, t)
            if m:
                sn_hits.add(m.group(1))
    if sn_hits:
        print(f"\n  Serial number(s) detected: {sn_hits}")
        for sn in sn_hits:
            print(f"\n  Set env vars:")
            print(f"    export ECOFLOW_DEVICE_IP={HOST}")
            print(f"    export ECOFLOW_DEVICE_SN={sn}")
            print(f"    export ECOFLOW_MQTT_USER={sn}")
            print(f"    export ECOFLOW_MQTT_PASS=<password>")
else:
    print("  No MQTT messages received on any attempt.")
    print()
    print("  The PowerStream broker requires the serial-number + local password.")
    print("  To obtain the local password:")
    print("    1. Open the EcoFlow app → select PowerStream → gear icon")
    print("    2. Look for 'Local Control' or 'MQTT Settings'")
    print("    3. Enable local API access to reveal the password")
    print()
    print(f"  Once you have them, run:")
    print(f"    python3 {sys.argv[0]} <DEVICE_SN> <LOCAL_PASSWORD>")
    print()
    print(f"  Then start the MCP server with:")
    print(f"    export ECOFLOW_DEVICE_IP={HOST}")
    print(f"    export ECOFLOW_DEVICE_SN=<SN>")
    print(f"    export ECOFLOW_MQTT_PASS=<password>")
print(SEP)
