"""Process limits, atomic output and available memory on Windows and Linux."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def configure_threads(threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    os.environ.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")


def replace(temporary, destination):
    deadline = time.monotonic() + 10
    while True:
        try:
            Path(temporary).replace(destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_memory():
    if os.name == "nt":
        import ctypes
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong) for key in ("total", "available", "total_page", "available_page", "total_virtual", "available_virtual", "extended")]
        memory = Memory(); memory.length = ctypes.sizeof(memory)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            raise OSError("Cannot query available memory")
        return memory.available
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        values = dict(line.split(":", 1) for line in meminfo.read_text().splitlines())
        free = int(values["MemAvailable"].split()[0]) * 1024
        # Respect cgroup v2 limits when running under a Linux container.
        limit, used = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
        if limit.exists() and used.exists() and limit.read_text().strip() != "max":
            free = min(free, max(0, int(limit.read_text()) - int(used.read_text())))
        return free
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def source_hashes():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*"))
            if p.suffix in {".py", ".json"}}


def stop_process_tree(process):
    """Stop only a worker tree that this runner created, including Windows venv launchers."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        result = subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode and process.poll() is None:
            raise RuntimeError(f"Cannot stop worker process tree {process.pid}")
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()
