"""The air-gap is process-wide and permanent, so each check runs in a fresh interpreter."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(code: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_guard_blocks_outbound_and_allows_loopback():
    out = run("""
        import json, socket, urllib.request
        from airlock import airgap
        airgap.install()
        res = {}
        def attempt(name, fn):
            try:
                fn(); res[name] = "allowed"
            except airgap.AirgapViolation:
                res[name] = "blocked"
            except Exception as e:  # urllib wraps the violation
                res[name] = "blocked" if "air-gap" in repr(e) else f"error: {e!r}"
        attempt("tcp_ip", lambda: socket.create_connection(("93.184.216.34", 443), timeout=2))
        attempt("dns", lambda: socket.getaddrinfo("api.openai.com", 443))
        attempt("udp", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("8.8.8.8", 53)))
        try:
            s6 = socket.socket(socket.AF_INET6)
            attempt("ipv6", lambda: s6.connect(("2606:4700:4700::1111", 443)))
        except OSError:  # host without IPv6: check the hook's decision directly
            attempt("ipv6", lambda: airgap._hook("socket.connect", (None, ("2606:4700:4700::1111", 443, 0, 0))))
        attempt("https_lib", lambda: urllib.request.urlopen("https://huggingface.co", timeout=2))

        srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(); port = srv.getsockname()[1]
        attempt("loopback_ip", lambda: socket.create_connection(("127.0.0.1", port), timeout=2).close())
        attempt("loopback_name", lambda: socket.create_connection(("localhost", port), timeout=2).close())
        res["self_test"] = airgap.self_test()
        res["blocked_hosts"] = sorted({b["host"] for b in airgap.status()["blocked_attempts"]})
        print(json.dumps(res))
    """)
    for k in ("tcp_ip", "dns", "udp", "ipv6", "https_lib"):
        assert out[k] == "blocked", (k, out[k])
    assert out["loopback_ip"] == "allowed" and out["loopback_name"] == "allowed"
    assert out["self_test"] is True
    assert "api.openai.com" in out["blocked_hosts"]
    assert "1.1.1.1" not in out["blocked_hosts"]  # self-test probe isn't reported as a phone-home


def test_guard_cannot_be_removed_and_install_is_idempotent():
    out = run("""
        import json, socket, sys
        from airlock import airgap
        for _ in range(5):
            airgap.install()
        hooks_installed = airgap.status()["installed"]
        try:
            socket.create_connection(("1.0.0.1", 80), timeout=1); r = "allowed"
        except airgap.AirgapViolation:
            r = "blocked"
        print(json.dumps({"installed": hooks_installed, "result": r, "blocked": len(airgap.status()["blocked_attempts"])}))
    """)
    assert out == {"installed": True, "result": "blocked", "blocked": 1}


def test_offline_env_and_no_proxy_for_loopback():
    out = run("""
        import json, os
        os.environ["NO_PROXY"] = "corp.internal"
        from airlock import airgap
        airgap.install()
        print(json.dumps({k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "ANONYMIZED_TELEMETRY", "NO_PROXY")}))
    """)
    assert out["HF_HUB_OFFLINE"] == "1" and out["ANONYMIZED_TELEMETRY"] == "False"
    assert out["NO_PROXY"].split(",") == ["corp.internal", "localhost", "127.0.0.1", "::1"]


def test_proxy_variables_are_removed_so_a_local_proxy_cannot_tunnel_out():
    out = run("""
        import json, os
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"
        os.environ["http_proxy"] = "http://127.0.0.1:3128"
        from airlock import airgap
        airgap.install()
        print(json.dumps({"left": [k for k in ("HTTPS_PROXY", "http_proxy") if k in os.environ],
                          "removed": airgap.status()["proxy_vars_removed"]}))
    """)
    assert out["left"] == [] and {"HTTPS_PROXY", "http_proxy"} <= set(out["removed"])


def test_strict_mode_limits_loopback_to_listed_ports():
    out = run("""
        import json, os, socket
        a = socket.socket(); a.bind(("127.0.0.1", 0)); a.listen()
        b = socket.socket(); b.bind(("127.0.0.1", 0)); b.listen()
        os.environ["AIRLOCK_LOOPBACK_PORTS"] = str(a.getsockname()[1])
        from airlock import airgap
        airgap.install()
        res = {}
        for name, sock in (("listed", a), ("other", b)):
            try:
                socket.create_connection(sock.getsockname(), timeout=1).close(); res[name] = "allowed"
            except airgap.AirgapViolation:
                res[name] = "blocked"
        print(json.dumps(res))
    """)
    assert out == {"listed": "allowed", "other": "blocked"}


def test_extra_allowed_host_for_on_prem_gpu_box():
    out = run("""
        import json, os
        os.environ["AIRLOCK_ALLOW_HOSTS"] = "10.0.0.5"
        from airlock import airgap
        airgap.install()
        print(json.dumps({"lan": airgap.is_local("10.0.0.5"), "other": airgap.is_local("10.0.0.6")}))
    """)
    assert out == {"lan": True, "other": False}
