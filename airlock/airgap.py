"""Process-wide network air-gap.

Installs a CPython audit hook (PEP 578) that vetoes every outbound socket
connection, UDP send and DNS lookup whose destination is not a loopback
address. Because audit hooks run inside the interpreter for *every* socket
call, this covers all Python code in the process: LangChain, httpx, requests,
ChromaDB telemetry, Hugging Face Hub, Streamlit - without trusting any of
them to honour an "offline" flag.

Audit hooks cannot be removed once added, so after `install()` the process
stays air-gapped for its whole lifetime. That is the point.

Proxy environment variables are removed from the process, because a
forwarding proxy listening on loopback would otherwise be a tunnel out. For the
paranoid, AIRLOCK_LOOPBACK_PORTS=11434 additionally restricts loopback traffic
to the listed ports (e.g. only Ollama).

Scope: this guards the Python process. Child processes (e.g. a shell command)
are not covered; use an OS firewall if you need defence in depth.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field

_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}

# Environment flags that ask libraries to stay offline / not phone home.
# The audit hook enforces this regardless; these just keep logs quiet.
OFFLINE_ENV = {
    "ANONYMIZED_TELEMETRY": "False",          # ChromaDB
    "HF_HUB_OFFLINE": "1",                    # Hugging Face Hub
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_DATASETS_OFFLINE": "1",
    "DO_NOT_TRACK": "1",
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "TOKENIZERS_PARALLELISM": "false",
}


class AirgapViolation(ConnectionRefusedError):
    """Raised when code in this process tries to reach a non-local host."""


_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@dataclass
class _State:
    installed: bool = False
    allowed_hosts: set[str] = field(default_factory=set)
    loopback_ports: set[int] | None = None      # strict mode: only these local ports
    removed_proxy_vars: set[str] = field(default_factory=set)
    blocked: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = _State()
_tls = threading.local()


def _host_of(address) -> str | None:
    """Extract the host part of a socket address; None for unix sockets."""
    if isinstance(address, (str, bytes)):
        return None  # AF_UNIX path - local by definition
    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, bytes):
            host = host.decode("ascii", "ignore")
        return host
    return None


def is_local(host) -> bool:
    if host is None or host == "":
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    host = str(host).strip("[]").split("%")[0].lower()
    if host in _LOCAL_NAMES or host in _state.allowed_hosts:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _record(event: str, host, port) -> None:
    if getattr(_tls, "self_test", False):
        return  # don't report our own probe as a phone-home attempt
    with _state.lock:
        _state.blocked.append({"event": event, "host": str(host), "port": port, "time": time.time()})
        del _state.blocked[:-100]  # keep the last 100 only


def _hook(event: str, args) -> None:
    # Keep this function fast: it runs for every audited event in the process.
    if not event.startswith("socket."):
        return
    if event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
        if len(args) < 2:
            return
        host = _host_of(args[1])
        port = args[1][1] if isinstance(args[1], tuple) and len(args[1]) > 1 else None
        if not is_local(host):
            _record(event, host, port)
            raise AirgapViolation(f"Airlock air-gap: blocked outbound connection to {host}:{port}")
        ports = _state.loopback_ports
        if ports is not None and host is not None and port not in ports and str(host) not in _state.allowed_hosts:
            _record(event, host, port)
            raise AirgapViolation(f"Airlock air-gap (strict): blocked local port {port}")
    elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                   "socket.gethostbyaddr"):
        host = args[0] if args else None
        if not is_local(host):
            _record(event, host, None)
            raise AirgapViolation(f"Airlock air-gap: blocked DNS lookup for {host!r}")


def install(extra_allowed_hosts: list[str] | None = None) -> None:
    """Air-gap this process. Idempotent; safe to call on every Streamlit rerun."""
    for key, value in OFFLINE_ENV.items():
        os.environ.setdefault(key, value)
    # A forwarding proxy listening on loopback would be a tunnel to the internet,
    # so stop libraries from routing anything through one...
    for key in _PROXY_VARS:
        if os.environ.pop(key, None) is not None:
            _state.removed_proxy_vars.add(key)
    # ...and make sure local model traffic never tries to.
    for key in ("NO_PROXY", "no_proxy"):
        current = [h for h in os.environ.get(key, "").split(",") if h.strip()]
        for h in ("localhost", "127.0.0.1", "::1"):
            if h not in current:
                current.append(h)
        os.environ[key] = ",".join(current)
    extra = list(extra_allowed_hosts or [])
    env_extra = os.environ.get("AIRLOCK_ALLOW_HOSTS", "")
    extra += [h.strip() for h in env_extra.split(",") if h.strip()]
    ports = os.environ.get("AIRLOCK_LOOPBACK_PORTS", "").strip()
    with _state.lock:
        _state.allowed_hosts.update(h.lower() for h in extra)
        if ports:
            _state.loopback_ports = {int(p) for p in ports.split(",") if p.strip()}
        if _state.installed:
            return
        sys.addaudithook(_hook)
        _state.installed = True


def status() -> dict:
    with _state.lock:
        ports = sorted(_state.loopback_ports) if _state.loopback_ports is not None else "any"
        return {
            "installed": _state.installed,
            "allowed": ["loopback (127.0.0.0/8, ::1, localhost)", *sorted(_state.allowed_hosts)],
            "loopback_ports": ports,
            "proxy_vars_removed": sorted(_state.removed_proxy_vars),
            "blocked_attempts": list(_state.blocked),
        }


def self_test() -> bool:
    """Prove the guard is live: an outbound connect must be refused."""
    if not _state.installed:
        return False
    _tls.self_test = True
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=0.5)
    except AirgapViolation:
        return True
    except OSError:
        return False
    finally:
        _tls.self_test = False
    return False
