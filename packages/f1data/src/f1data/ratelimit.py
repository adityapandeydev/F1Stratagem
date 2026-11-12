"""Honour FastF1's request limits across waits and process restarts.

FastF1 enforces its hourly limits client-side with an in-memory sliding window of
call timestamps (one per call, including calls that were refused). A new process
starts with an empty window, so a restarted download could exceed the real hourly
budget. This module persists those timestamps in the data root, restores them on
start-up, and computes how long to wait until a useful budget is free again.

It reads FastF1 internals defensively: if their shape changes, persistence is
skipped and callers fall back to a fixed wait.
"""

import json
import time

STATE_FILE = "rate-limit-state.json"
# Calls to have available before resuming, so a resumed session is unlikely to be
# interrupted again a few requests later. A session load uses roughly 10-15 calls.
RESUME_BUDGET = 100
SAFETY_SECONDS = 30


def limiters():
    """FastF1's hourly call-count limiters keyed by their description."""
    try:
        from fastf1.req import _CallsPerIntervalLimitRaise, _SessionWithRateLimiting
        found = {}
        for group in _SessionWithRateLimiting._RATE_LIMITS.values():
            for limiter in group:
                if isinstance(limiter, _CallsPerIntervalLimitRaise):
                    found[limiter._info] = limiter
        return found
    except (ImportError, AttributeError):
        return {}


def save(root, now=None):
    now = time.time() if now is None else now
    state = {info: [t for t in limiter._timestamps if t > now - limiter._interval]
             for info, limiter in limiters().items()}
    path = root / STATE_FILE
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"saved_at": now, "timestamps": state}), encoding="utf-8")
    temporary.replace(path)


def restore(root, now=None):
    """Load timestamps from earlier processes that are still inside the window."""
    now = time.time() if now is None else now
    path = root / STATE_FILE
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))["timestamps"]
    except (OSError, ValueError, KeyError):
        return 0
    restored = 0
    for info, limiter in limiters().items():
        recent = sorted(t for t in saved.get(info, []) if t > now - limiter._interval)
        merged = sorted(set(recent) | set(limiter._timestamps))
        limiter._timestamps.clear()
        limiter._timestamps.extend(merged[-limiter._timestamps.maxlen:])
        restored += len(recent)
    return restored


def seconds_until_budget(budget=RESUME_BUDGET, now=None):
    """Seconds until every limiter has at least `budget` free calls; None if unknown."""
    now = time.time() if now is None else now
    found = limiters()
    if not found:
        return None
    wait = 0.0
    for limiter in found.values():
        capacity = limiter._timestamps.maxlen
        window = sorted(t for t in limiter._timestamps if t > now - limiter._interval)
        need = min(budget, capacity) - (capacity - len(window))
        if need > 0:
            wait = max(wait, window[need - 1] + limiter._interval - now)
    return wait + SAFETY_SECONDS if wait > 0 else 0.0
