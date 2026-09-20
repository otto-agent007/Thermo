"""Optional CPU work accounting, excluded from scientific request/result identity."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_ACTIVE = ContextVar("thermo_runtime_work", default=None)


@contextmanager
def collect_work():
    ledger = {}
    token = _ACTIVE.set((ledger, []))
    try:
        yield ledger
    finally:
        _ACTIVE.reset(token)


def timed_work(name):
    """Record inclusive and exclusive CPU wall time; preserve values and exceptions."""

    def decorate(function):
        @wraps(function)
        def call(*args, **kwargs):
            active = _ACTIVE.get()
            if active is None:
                return function(*args, **kwargs)
            ledger, stack = active
            frame = [0.0]
            stack.append(frame)
            start = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter() - start
                stack.pop()
                if stack:
                    stack[-1][0] += elapsed
                row = ledger.setdefault(
                    name, {"calls": 0, "inclusive_seconds": 0.0, "exclusive_seconds": 0.0}
                )
                row["calls"] += 1
                row["inclusive_seconds"] += elapsed
                row["exclusive_seconds"] += max(0.0, elapsed - frame[0])

        return call

    return decorate
