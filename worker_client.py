# worker_client.py — Nuke-side manager for the out-of-process SAM worker.
#
# WHY DETACHED: endpoint-security (EDR) tooling in some environments blocks
# torch's c10.dll initialization (WinError 1114) in ANY descendant of
# Nuke.exe. Proven by elimination: identical env and module lists, torch
# loads from PowerShell and from a WMI-detached process, dies in every child
# of Nuke regardless of env/console/cwd/job-breakaway. So on Windows the
# worker is spawned OUTSIDE Nuke's process tree via WMI (parent becomes
# WmiPrvSE.exe) and reached over a token-protected localhost socket
# advertised in a rendezvous file. On POSIX a plain piped child is used.
#
# One persistent worker per Nuke session: the SAM model stays warm on the
# GPU between clicks. The worker self-destructs after 30 min idle, so a
# crashed Nuke can't leak workers.

import atexit
import collections
import json
import os
import platform
import queue
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

PACKAGE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
WORKER_SCRIPT = PACKAGE_DIR / "worker.py"

IS_WINDOWS = platform.system() == "Windows"
CREATE_NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

_lock = threading.Lock()
_busy = False

# socket mode (Windows)
_sock = None
_sock_queue = None
_worker_pid = None

# pipe mode (POSIX fallback)
_proc = None
_out_queue = None
_stderr_tail = collections.deque(maxlen=60)


class WorkerCancelled(RuntimeError):
    """User pressed cancel on the progress bar."""


def _worker_python() -> str:
    """Path to the venv's own Python (where torch is known to load)."""
    from . import env_bootstrap

    venv = env_bootstrap.get_venv_path()
    exe = (
        venv / "Scripts" / "python.exe" if IS_WINDOWS else venv / "bin" / "python"
    )
    if not exe.exists():
        raise RuntimeError(
            f"Worker python not found: {exe}\n"
            f"Run install.py from a terminal first."
        )
    return str(exe)


def _rdv_path() -> str:
    """Per-Nuke-session rendezvous file (worker advertises port/token here)."""
    return os.path.join(
        tempfile.gettempdir(), f"h2_samvit_rdv_{os.getpid()}.json"
    )


def _log_tail(n=15) -> str:
    try:
        with open(_rdv_path() + ".log", "r", encoding="utf-8", errors="replace") as f:
            return "\n".join(f.read().splitlines()[-n:])
    except Exception:
        return "(no worker log found)"


def _progress_task(label):
    """A nuke.ProgressTask in GUI sessions, or a no-op stand-in otherwise."""
    try:
        import nuke

        if nuke.GUI:
            return nuke.ProgressTask(label)
    except Exception:
        pass

    class _Null:
        def setMessage(self, *_):
            pass

        def setProgress(self, *_):
            pass

        def isCancelled(self):
            return False

    return _Null()


# ──────────────────────────────────────────────────────────────────────
#  Windows: detached spawn (WMI) + localhost socket
# ──────────────────────────────────────────────────────────────────────

def _spawn_detached(rdv: str) -> None:
    """Create the worker OUTSIDE Nuke's process tree via WMI.

    pythonw.exe is used so the detached worker gets no console window.
    """
    py = _worker_python()
    pyw = str(Path(py).with_name("pythonw.exe"))
    exe = pyw if os.path.exists(pyw) else py

    cmd = f"{exe} {WORKER_SCRIPT} --serve {rdv}"
    ps = (
        "Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        "-Arguments @{CommandLine='" + cmd + "'} | "
        "Select-Object -ExpandProperty ReturnValue"
    )
    r = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=90,
        creationflags=CREATE_NO_WINDOW,
    )
    rc = (r.stdout or "").strip()
    if r.returncode != 0 or rc not in ("0",):
        raise RuntimeError(
            f"Detached worker spawn failed (WMI rc={rc or '?'}).\n"
            f"{(r.stderr or '').strip()[-400:]}"
        )
    print("[H2 SamViT] Worker spawn requested (detached via WMI)")


def _sock_reader(sock, q):
    buf = b""
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, _, buf = buf.partition(b"\n")
                q.put(line.decode("utf-8", "replace"))
    except OSError:
        pass
    q.put(None)  # EOF sentinel


def _kill_pid(pid) -> None:
    if not pid:
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=20,
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        pass


def _connect_socket(timeout=240):
    """Wait for the rendezvous file, connect + authenticate. Locked caller."""
    global _sock, _sock_queue, _worker_pid

    rdv = _rdv_path()
    task = _progress_task("H2 SamViT — starting worker")
    task.setMessage("Starting inference worker (loading torch)…")
    started = time.time()
    beat = 0
    try:
        while not os.path.exists(rdv):
            if task.isCancelled():
                raise WorkerCancelled("Cancelled while starting the worker.")
            if time.time() - started > timeout:
                raise RuntimeError(
                    "Inference worker did not start.\n\nWorker log:\n" + _log_tail()
                )
            beat = (beat + 3) % 100
            task.setProgress(beat)
            time.sleep(0.3)
    finally:
        del task

    with open(rdv, "r") as f:
        info = json.load(f)

    s = socket.create_connection(("127.0.0.1", int(info["port"])), timeout=15)
    s.sendall((json.dumps({"token": info["token"]}) + "\n").encode("utf-8"))
    s.settimeout(None)

    q = queue.Queue()
    threading.Thread(target=_sock_reader, args=(s, q), daemon=True).start()

    hello = q.get(timeout=30)
    if hello is None or not json.loads(hello).get("ok"):
        raise RuntimeError("Worker handshake failed.\n\n" + _log_tail())

    _sock, _sock_queue, _worker_pid = s, q, info.get("pid")
    print(f"[H2 SamViT] Connected to worker (pid {_worker_pid})")


def _ensure_socket_worker():
    """Reuse a live worker, reconnect to a surviving one, or spawn fresh."""
    global _sock, _sock_queue
    if _sock is not None:
        return
    rdv = _rdv_path()
    if os.path.exists(rdv):
        try:
            _connect_socket(timeout=5)  # worker from a previous module load
            return
        except Exception:
            try:
                os.remove(rdv)
            except OSError:
                pass
    _spawn_detached(rdv)
    _connect_socket()


# ──────────────────────────────────────────────────────────────────────
#  POSIX fallback: plain piped child
# ──────────────────────────────────────────────────────────────────────

def _pump_stdout(pipe, q):
    for line in iter(pipe.readline, ""):
        q.put(line)
    q.put(None)


def _pump_stderr(pipe):
    for line in iter(pipe.readline, ""):
        line = line.rstrip()
        _stderr_tail.append(line)
        print(line)


def _ensure_pipe_worker():
    global _proc, _out_queue
    if _proc is not None and _proc.poll() is None:
        return
    _proc = subprocess.Popen(
        [_worker_python(), "-u", str(WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PACKAGE_DIR),
    )
    _out_queue = queue.Queue()
    threading.Thread(
        target=_pump_stdout, args=(_proc.stdout, _out_queue), daemon=True
    ).start()
    threading.Thread(target=_pump_stderr, args=(_proc.stderr,), daemon=True).start()
    print(f"[H2 SamViT] Worker started (pid {_proc.pid})")


# ──────────────────────────────────────────────────────────────────────
#  Shared request path
# ──────────────────────────────────────────────────────────────────────

def _send_line(payload: dict) -> None:
    data = json.dumps(payload)
    if IS_WINDOWS:
        _sock.sendall((data + "\n").encode("utf-8"))
    else:
        _proc.stdin.write(data + "\n")
        _proc.stdin.flush()


def _kill_worker() -> None:
    global _sock, _sock_queue, _proc
    if IS_WINDOWS:
        if _sock is not None:
            try:
                _sock.close()
            except OSError:
                pass
        _kill_pid(_worker_pid)
        _sock = None
        _sock_queue = None
        try:
            os.remove(_rdv_path())
        except OSError:
            pass
    else:
        if _proc is not None:
            _proc.kill()
        _proc = None


def _death_report() -> str:
    tail = _log_tail() if IS_WINDOWS else "\n".join(list(_stderr_tail)[-12:])
    return f"Inference worker exited unexpectedly.\n\nLast output:\n{tail}"


def request(payload, label="H2 SamViT", timeout=None):
    """Send one request to the worker and wait for its JSON response.

    Runs a Nuke progress bar while waiting; cancelling it kills the worker
    (a torch model load can't be interrupted any other way). Raises on
    worker error, death, cancel, or timeout.
    """
    global _busy
    with _lock:
        if _busy:
            raise RuntimeError(
                "An inference is already running — wait for it to finish."
            )
        if IS_WINDOWS:
            _ensure_socket_worker()
            q = _sock_queue
        else:
            _ensure_pipe_worker()
            q = _out_queue
        _busy = True
        try:
            _send_line(payload)
        except Exception:
            _busy = False
            _kill_worker()
            raise

    task = _progress_task(label)
    task.setMessage("Waiting for inference worker…")
    started = time.time()
    beat = 0

    try:
        while True:
            if task.isCancelled():
                _kill_worker()
                raise WorkerCancelled("Cancelled by user.")
            if timeout and (time.time() - started) > timeout:
                _kill_worker()
                raise RuntimeError(f"Worker timed out after {timeout:.0f}s.")
            try:
                line = q.get(timeout=0.25)
            except queue.Empty:
                beat = (beat + 3) % 100
                task.setProgress(beat)
                elapsed = int(time.time() - started)
                task.setMessage(f"Inference worker running… {elapsed}s")
                continue

            if line is None:  # EOF — worker died / connection lost
                report = _death_report()
                _kill_worker()
                raise RuntimeError(report)
            try:
                resp = json.loads(line)
            except Exception:
                continue  # stray non-JSON line — skip

            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "unknown worker error"))
            return resp
    finally:
        _busy = False
        del task  # closes the progress bar


def ping(timeout=300):
    """Verify the worker starts and torch loads in it. Returns the response."""
    return request({"cmd": "ping"}, label="H2 SamViT — starting worker", timeout=timeout)


def clear():
    """Free the worker's models/VRAM (keeps the process alive)."""
    alive = (_sock is not None) if IS_WINDOWS else (
        _proc is not None and _proc.poll() is None
    )
    if not alive:
        return
    try:
        request({"cmd": "clear"}, label="H2 SamViT — clearing models", timeout=60)
    except Exception as e:
        print(f"[H2 SamViT] clear failed ({e}); restarting worker instead")
        _kill_worker()


def shutdown():
    """Stop the worker (used on Nuke exit)."""
    try:
        alive = (_sock is not None) if IS_WINDOWS else (
            _proc is not None and _proc.poll() is None
        )
        if alive:
            try:
                _send_line({"cmd": "quit"})
                time.sleep(0.5)
            except Exception:
                pass
    finally:
        _kill_worker()


atexit.register(shutdown)
