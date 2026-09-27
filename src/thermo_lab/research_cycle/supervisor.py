"""Private Linux command guardian, independent of the worker parent's lifetime.

Launched by absolute file path so no checkout imports run before ownership is
recorded. Both guardian and child inherit the worker's advisory lock.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import sys
import time
from pathlib import Path


def persist(path: Path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    # Adopt and reap descendants if a command dies before its children. This
    # avoids confusing dead zombie groups with live ownership during recovery.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot establish child subreaper")
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            stat = Path("/proc/self/stat").read_text().split(") ", 1)[1].split()
            persist(
                config_path.parent / "owner.json",
                {
                    "pid": os.getpid(),
                    "pgid": os.getpgrp(),
                    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                    "start_ticks": stat[19],
                },
            )
            os.execvpe(config["argv"][0], config["argv"], os.environ)
        except BaseException:
            os._exit(127)
    status = None
    timed_out = False
    try:
        while status is None and not stopped:
            found, raw = os.waitpid(pid, os.WNOHANG)
            if found:
                status = raw
                break
            if time.monotonic() >= config["deadline_monotonic"]:
                timed_out = True
                break
            time.sleep(0.02)
    finally:
        try:
            os.killpg(pid, signal.SIGTERM)
            time.sleep(0.05)
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            # A deadline may fire before the forked child calls setsid. An
            # unreaped direct child cannot have its PID reused.
            if status is None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if status is None:
            _, status = os.waitpid(pid, 0)
        # The trusted adapter does not create separate sessions. Reap only
        # descendants in this command's group, with a bounded cleanup window.
        until = time.monotonic() + 0.5
        while time.monotonic() < until:
            try:
                found, _ = os.waitpid(-pid, os.WNOHANG)
                if not found:
                    time.sleep(0.01)
            except ChildProcessError:
                break
        try:
            os.killpg(pid, 0)
            exited = False
        except ProcessLookupError:
            exited = True
        persist(
            config_path.parent / "guardian-result.json",
            {
                "exit_status": os.waitstatus_to_exitcode(status),
                "timed_out": timed_out,
                "group_exited": exited,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
