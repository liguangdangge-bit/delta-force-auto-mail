from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import psutil


DELTA_FORCE_GAME_PROCESS = "deltaforceclient-win64-shipping.exe"
DELTA_FORCE_LAUNCHER_PROCESS = "delta_force_launcher.exe"


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    parent_pid: int
    name: str
    executable: str
    created_at: float | None
    status: str
    depth: int = 0


@dataclass(frozen=True)
class ProcessTreeTermination:
    selection_source: str
    root_pid: int
    root_created_at: float | None
    processes: tuple[ProcessRecord, ...]
    kill_attempted_pids: tuple[int, ...]
    kill_errors: dict[int, str]


class GameProcessSelectionError(RuntimeError):
    pass


def _safe_process_record(process: psutil.Process, *, depth: int = 0) -> ProcessRecord:
    try:
        with process.oneshot():
            pid = int(process.pid)
            parent_pid = int(process.ppid())
            name = process.name()
            try:
                executable = process.exe()
            except (psutil.AccessDenied, psutil.ZombieProcess, OSError):
                executable = ""
            try:
                created_at = float(process.create_time())
            except (psutil.AccessDenied, psutil.ZombieProcess, OSError):
                created_at = None
            try:
                status = process.status()
            except (psutil.AccessDenied, psutil.ZombieProcess, OSError):
                status = "unknown"
    except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
        raise GameProcessSelectionError(f"进程 {process.pid} 已经退出") from exc
    return ProcessRecord(
        pid=pid,
        parent_pid=parent_pid,
        name=name,
        executable=executable,
        created_at=created_at,
        status=status,
        depth=depth,
    )


def _process_name(process: psutil.Process) -> str:
    try:
        return process.name().strip().casefold()
    except (psutil.Error, OSError):
        return ""


def _window_process_id(hwnd: int) -> int | None:
    if os.name != "nt" or not hwnd:
        return None
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    pid = wintypes.DWORD()
    if not user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid)):
        return None
    return int(pid.value) or None


def find_game_root_process(preferred_window_handle: int | None = None) -> tuple[psutil.Process, str]:
    window_pid = _window_process_id(preferred_window_handle or 0)
    if window_pid is not None:
        try:
            process = psutil.Process(window_pid)
            if _process_name(process) == DELTA_FORCE_GAME_PROCESS:
                return process, "bound_game_window"
        except (psutil.Error, OSError):
            pass

    candidates = [
        process
        for process in psutil.process_iter()
        if _process_name(process) == DELTA_FORCE_GAME_PROCESS
    ]
    if not candidates:
        raise GameProcessSelectionError("没有找到正在运行的三角洲游戏主进程")
    if len(candidates) > 1:
        pids = ", ".join(str(process.pid) for process in candidates)
        raise GameProcessSelectionError(
            f"找到多个三角洲游戏主进程（PID {pids}），为避免误结束已取消操作"
        )
    return candidates[0], "unique_process_name"


def _tree_records(root: psutil.Process) -> tuple[list[ProcessRecord], dict[int, psutil.Process]]:
    processes: dict[int, psutil.Process] = {root.pid: root}
    parents: dict[int, int] = {}
    try:
        descendants = root.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    for child in descendants:
        processes[child.pid] = child
        try:
            parents[child.pid] = child.ppid()
        except (psutil.Error, OSError):
            parents[child.pid] = root.pid

    depths: dict[int, int] = {root.pid: 0}

    def resolve_depth(pid: int, seen: set[int] | None = None) -> int:
        if pid in depths:
            return depths[pid]
        active_seen = set() if seen is None else set(seen)
        if pid in active_seen:
            return 1
        active_seen.add(pid)
        parent_pid = parents.get(pid, root.pid)
        depth = resolve_depth(parent_pid, active_seen) + 1 if parent_pid in processes else 1
        depths[pid] = depth
        return depth

    records: list[ProcessRecord] = []
    for pid, process in processes.items():
        try:
            records.append(_safe_process_record(process, depth=resolve_depth(pid)))
        except GameProcessSelectionError:
            continue
    records.sort(key=lambda item: (item.depth, item.pid))
    return records, processes


def _same_process_still_running(record: ProcessRecord) -> bool:
    try:
        process = psutil.Process(record.pid)
        if record.created_at is not None:
            return abs(process.create_time() - record.created_at) < 0.001
        return process.is_running()
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, OSError):
        # An unreadable process has not been proven to have exited.
        return True


def process_records_still_running(
    records: list[ProcessRecord] | tuple[ProcessRecord, ...],
) -> list[ProcessRecord]:
    return [record for record in records if _same_process_still_running(record)]


class ProcessExitCleanup:
    """Bounded retries for the exact processes selected by the initial kill."""

    def __init__(self, records, started_at: float) -> None:
        self.records = tuple(records)
        self.next_retry_at = started_at + 5.0
        self.attempts = 0

    def poll(self, now: float) -> tuple[list[ProcessRecord], dict | None]:
        remaining = process_records_still_running(self.records)
        if not remaining or self.attempts >= 3 or now < self.next_retry_at:
            return remaining, None
        self.attempts += 1
        self.next_retry_at = now + 5.0
        attempted = []
        errors = {}
        for record in sorted(remaining, key=lambda item: (item.depth, item.pid), reverse=True):
            try:
                if record.created_at is None:
                    errors[record.pid] = "缺少创建时间，不能安全重试结束"
                    continue
                process = psutil.Process(record.pid)
                if (abs(process.create_time() - record.created_at) >= 0.001
                        or process.name().casefold() != record.name.casefold()):
                    continue
                attempted.append(record.pid)
                process.kill()
            except psutil.NoSuchProcess:
                continue
            except (psutil.Error, OSError) as exc:
                errors[record.pid] = f"{type(exc).__name__}: {exc}"
        # Sending a kill request is not proof of exit. Recheck even if kill()
        # returned successfully (Windows termination can remain pending).
        remaining = process_records_still_running(self.records)
        return remaining, {
            "attempt": self.attempts,
            "attempted_pids": attempted,
            "remaining_pids": [record.pid for record in remaining],
            "errors": errors,
        }


def terminate_game_process_tree(
    preferred_window_handle: int | None = None,
) -> ProcessTreeTermination:
    root, source = find_game_root_process(preferred_window_handle)
    root_record = _safe_process_record(root)
    tree, process_map = _tree_records(root)
    attempted: list[int] = []
    errors: dict[int, str] = {}

    for record in sorted(tree, key=lambda item: (item.depth, item.pid), reverse=True):
        process = process_map.get(record.pid)
        if process is None or not _same_process_still_running(record):
            continue
        attempted.append(record.pid)
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, psutil.Error, OSError) as exc:
            errors[record.pid] = str(exc)

    return ProcessTreeTermination(
        selection_source=source,
        root_pid=root_record.pid,
        root_created_at=root_record.created_at,
        processes=tuple(tree),
        kill_attempted_pids=tuple(attempted),
        kill_errors=errors,
    )
