import csv
import os
import threading
import time
from collections import deque
from typing import Any

import psutil


FEATURE_COLUMNS = [
    "timestamp",
    "cpu_usage",
    "memory_usage",
    "disk_read",
    "disk_write",
    "network_in",
    "network_out",
]


class RealtimeTelemetryCollector:
    """Collects live OS telemetry using psutil once per second."""

    def __init__(self, data_dir: str, interval: float = 1.0, max_rows: int = 2000):
        self.data_dir = data_dir
        self.csv_path = os.path.join(data_dir, "telemetry.csv")
        self.interval = interval
        self.max_rows = max_rows

        self._history: deque[dict[str, float]] = deque(maxlen=max_rows)
        self._processes: list[dict[str, Any]] = []
        self._total_processes = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process_thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._last_sample_time: float | None = None

        self._last_counter_time: float | None = None
        self._last_disk = None
        self._last_net = None

        os.makedirs(self.data_dir, exist_ok=True)
        self._load_existing_history()
        self._ensure_csv()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_sample_time(self) -> float | None:
        return self._last_sample_time

    def start(self) -> None:
        if self.running:
            return

        # Prime psutil CPU counters so the second sample is meaningful.
        psutil.cpu_percent(interval=None)
        for proc in psutil.process_iter():
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="telemetry-collector", daemon=True)
        self._process_thread = threading.Thread(target=self._run_processes, name="process-collector", daemon=True)
        self._thread.start()
        self._process_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2)

    def get_latest(self) -> dict[str, Any]:
        with self._lock:
            if self._history:
                return dict(self._history[-1])
        return self._empty_sample()

    def get_history(self, limit: int = 60) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 60), self.max_rows))
        with self._lock:
            return [dict(row) for row in list(self._history)[-safe_limit:]]

    def get_processes(self) -> dict[str, Any]:
        with self._lock:
            return {
                "processes": [dict(proc) for proc in self._processes],
                "total_processes": self._total_processes,
                "timestamp": time.time(),
            }

    def collect_once(self) -> dict[str, Any]:
        now = time.time()

        cpu = float(psutil.cpu_percent(interval=None))
        memory = float(psutil.virtual_memory().percent)
        disk_read, disk_write = self._disk_speeds(now)
        net_in, net_out = self._network_speeds(now)

        sample = {
            "timestamp": round(now, 3),
            "cpu_usage": round(cpu, 2),
            "memory_usage": round(memory, 2),
            "disk_read": round(disk_read, 2),
            "disk_write": round(disk_write, 2),
            "network_in": round(net_in, 2),
            "network_out": round(net_out, 2),
        }

        with self._lock:
            self._history.append(sample)
            self._last_sample_time = now
            self._write_csv_locked()

        return dict(sample)

    def refresh_processes(self) -> dict[str, Any]:
        processes, total = self._collect_processes()
        with self._lock:
            self._processes = processes
            self._total_processes = total
        return self.get_processes()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            try:
                self.collect_once()
                self._last_error = None
            except Exception as exc:  # Keep the monitor alive if one sample fails.
                self._last_error = str(exc)

            elapsed = time.time() - started
            self._stop_event.wait(max(0.05, self.interval - elapsed))

    def _run_processes(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            try:
                self.refresh_processes()
            except Exception as exc:
                self._last_error = str(exc)

            elapsed = time.time() - started
            self._stop_event.wait(max(0.05, self.interval - elapsed))

    def _disk_speeds(self, now: float) -> tuple[float, float]:
        counters = psutil.disk_io_counters()
        if counters is None:
            return 0.0, 0.0

        if self._last_disk is None or self._last_counter_time is None:
            self._last_disk = counters
            return 0.0, 0.0

        elapsed = max(now - self._last_counter_time, 0.001)
        read_kb = max(0.0, (counters.read_bytes - self._last_disk.read_bytes) / 1024 / elapsed)
        write_kb = max(0.0, (counters.write_bytes - self._last_disk.write_bytes) / 1024 / elapsed)
        self._last_disk = counters
        return read_kb, write_kb

    def _network_speeds(self, now: float) -> tuple[float, float]:
        counters = psutil.net_io_counters()
        if counters is None:
            return 0.0, 0.0

        if self._last_net is None or self._last_counter_time is None:
            self._last_net = counters
            self._last_counter_time = now
            return 0.0, 0.0

        elapsed = max(now - self._last_counter_time, 0.001)
        in_kb = max(0.0, (counters.bytes_recv - self._last_net.bytes_recv) / 1024 / elapsed)
        out_kb = max(0.0, (counters.bytes_sent - self._last_net.bytes_sent) / 1024 / elapsed)
        self._last_net = counters
        self._last_counter_time = now
        return in_kb, out_kb

    def _collect_processes(self) -> tuple[list[dict[str, Any]], int]:
        processes: list[dict[str, Any]] = []
        total = 0

        attrs = ["pid", "name", "status", "username", "memory_percent", "memory_info"]
        for proc in psutil.process_iter(attrs):
            try:
                info = proc.info
                pid = int(info.get("pid") or proc.pid)
                name = str(info.get("name") or "unknown")[:80]
                if pid == 0 or name.lower() == "system idle process":
                    continue

                total += 1
                memory_info = info.get("memory_info")
                mem_kb = float(getattr(memory_info, "rss", 0) or 0) / 1024
                memory_percent = float(info.get("memory_percent") or 0.0)
                cpu_percent = float(proc.cpu_percent(interval=None) or 0.0)
                status = str(info.get("status") or "unknown")

                processes.append(
                    {
                        "pid": pid,
                        "name": name,
                        "cpu": round(cpu_percent, 2),
                        "cpu_percent": round(cpu_percent, 2),
                        "mem": round(memory_percent, 2),
                        "memory_percent": round(memory_percent, 2),
                        "mem_kb": round(mem_kb, 2),
                        "status": status,
                        "state": status,
                        "user": str(info.get("username") or "unknown")[:40],
                        "username": str(info.get("username") or "unknown")[:40],
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        processes.sort(key=lambda item: (item["cpu_percent"], item["memory_percent"]), reverse=True)
        return processes[:15], total

    def _load_existing_history(self) -> None:
        if not os.path.exists(self.csv_path):
            return

        try:
            with open(self.csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            rows = []

        for row in rows[-self.max_rows :]:
            try:
                self._history.append(
                    {
                        "timestamp": float(row.get("timestamp", 0) or 0),
                        "cpu_usage": float(row.get("cpu_usage", 0) or 0),
                        "memory_usage": float(row.get("memory_usage", 0) or 0),
                        "disk_read": float(row.get("disk_read", 0) or 0),
                        "disk_write": float(row.get("disk_write", 0) or 0),
                        "network_in": float(row.get("network_in", 0) or 0),
                        "network_out": float(row.get("network_out", 0) or 0),
                    }
                )
            except (TypeError, ValueError):
                continue

    def _ensure_csv(self) -> None:
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()

    def _write_csv_locked(self) -> None:
        tmp_path = self.csv_path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            writer.writerows(self._history)
        os.replace(tmp_path, self.csv_path)

    @staticmethod
    def _empty_sample() -> dict[str, Any]:
        return {
            "timestamp": round(time.time(), 3),
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_read": 0.0,
            "disk_write": 0.0,
            "network_in": 0.0,
            "network_out": 0.0,
        }
