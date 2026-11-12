"""Historical session discovery and bounded, resumable ingestion orchestration."""

import numbers
import pandas as pd

from datetime import datetime, timezone
import time

from .sessions import ProviderAccessError, RateLimitReached, present, unresolved_attempt

# Schedule starts do not prove completion. Use a conservative six-hour holdback
# for historical batch discovery; ingestion still validates the source itself.
COMPLETION_HOLDBACK = pd.Timedelta(hours=6)

# Every session that sets a grid or awards a result. FastF1 names vary by season:
# 2021 "Sprint Qualifying" was the sprint race, 2023 "Sprint Shootout" and
# 2024+ "Sprint Qualifying" are the sprint grid sessions. Listing all of them
# keeps the selection correct without interpreting the season-specific meaning.
COMPETITIVE_SESSIONS = frozenset({"Qualifying", "Sprint Shootout", "Sprint Qualifying", "Sprint", "Race"})
TELEMETRY_SCOPES = {"all": None, "competitive": COMPETITIVE_SESSIONS}


def wants_telemetry(item, telemetry, scope=None):
    """Telemetry tier requested for this session; scope None means every session."""
    return bool(telemetry) and (scope is None or item.get("session_name") in scope)


def validate_year(year):
    if isinstance(year, bool) or not isinstance(year, numbers.Integral) or year < 2018 or year > 9999:
        raise ValueError("Detailed FastF1 ingestion requires an integer season from 2018 onward")


def select_schedule_sessions(schedule, year, weekends=None, now=None, purpose="consecutive-season corpus",
                             testing=False):
    """Testing events have RoundNumber 0 and are identified by their order in the
    season (FastF1's test_number), giving IDs such as 2023-T1-2."""
    validate_year(year)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("Discovery time must include a timezone")
    selected = None if weekends is None else {int(item["round"]): item for item in weekends}
    items, excluded, seen = [], [], set()
    test_number = 0
    for _, event in schedule.iterrows():
        round_number = int(event["RoundNumber"])
        is_testing = str(event.get("EventFormat", "")) == "testing"
        if is_testing:
            test_number += 1
            if not testing or selected is not None:
                continue
        elif round_number <= 0 or (selected is not None and round_number not in selected):
            continue
        if not is_testing:
            seen.add(round_number)
        choice = selected[round_number] if selected is not None else None
        if choice is not None and event["EventName"] != choice["event_name"]:
            raise ValueError(f"Event identity mismatch: {choice['event_name']} vs {event['EventName']}")
        for slot in range(1, 6):
            name = event.get(f"Session{slot}")
            if not isinstance(name, str) or not name.strip():
                continue
            date = pd.to_datetime(event.get(f"Session{slot}DateUtc"), utc=True, errors="coerce")
            item = {
                "year": int(year), "round": round_number, "slot": slot,
                "session_id": f"{year}-T{test_number}-{slot}" if is_testing else f"{year}-{round_number:02}-{slot}",
                "event_name": str(event["EventName"]), "session_name": name,
                "scheduled_start_utc": date.isoformat() if pd.notna(date) else None,
                "purpose": choice.get("purpose", purpose) if choice is not None else purpose,
                "location": str(event.get("Location", "")), "country": str(event.get("Country", "")),
                "event_format": str(event.get("EventFormat", "")),
            }
            if is_testing:
                item.update(is_testing=True, test_number=test_number)
            support = event.get("F1ApiSupport")
            reason = None
            if pd.isna(support) or not bool(support):
                reason = "api_unsupported"
            elif pd.isna(date):
                reason = "date_unknown"
            elif date + COMPLETION_HOLDBACK > now:
                reason = "not_yet_finished"
            if reason:
                excluded.append({**item, "reason": reason})
            else:
                items.append(item)
    if selected is not None and set(selected).difference(seen):
        raise ValueError(f"Rounds absent from source schedule: {sorted(set(selected).difference(seen))}")
    return sorted(items, key=lambda item: item["session_id"]), excluded


def ingestion_batch(items, root, telemetry, attempts, max_sessions=None, ingest_session=None, telemetry_scope=None,
                    rate_limit_wait=None, max_rate_limit_waits=6, sleep=time.sleep,
                    compute_wait=None, after_session=None, retry_failed=False):
    """rate_limit_wait (seconds): when set, a rate-limit stop pauses and retries the
    same session instead of ending the batch. compute_wait() may return the exact
    seconds until FastF1's window frees enough calls; rate_limit_wait is then the
    fallback when it returns None. max_rate_limit_waits caps consecutive waits on
    one session, so a limit that never clears still ends the run. after_session()
    runs after every attempt (used to persist limiter state). Sessions an earlier
    attempt left unresolved (see sessions.unresolved_attempt) are listed and skipped
    unless retry_failed is set."""
    from .sessions import ingest
    if max_sessions is not None and (not isinstance(max_sessions, int) or isinstance(max_sessions, bool) or max_sessions < 1):
        raise ValueError("max_sessions must be a positive integer")
    base_ingest = ingest_session or ingest

    def ingest_session(item, root, item_telemetry, attempts):
        waits = 0
        while True:
            try:
                return base_ingest(item, root, item_telemetry, attempts)
            except RateLimitReached:
                if rate_limit_wait is None or waits >= max_rate_limit_waits:
                    raise
                waits += 1
                exact = compute_wait() if compute_wait else None
                delay = max(60, int(exact)) if exact is not None else rate_limit_wait
                resume = datetime.now(timezone.utc).timestamp() + delay
                print(f"RATE LIMIT at {item['session_id']}; waiting {round(delay / 60)} min "
                      f"(until {datetime.fromtimestamp(resume).strftime('%H:%M')} local), wait {waits}/{max_rate_limit_waits}",
                      flush=True)
                sleep(delay)
            finally:
                if after_session:
                    after_session()

    result = {"attempted": 0, "skipped": 0, "succeeded": [], "failed": [], "deferred": [],
              "unresolved_skipped": {}, "stopped_reason": None}
    for index, item in enumerate(items):
        directory = root / "sessions" / str(item["year"]) / item["session_id"]
        item_telemetry = wants_telemetry(item, telemetry, telemetry_scope)
        complete = present(directory, item_telemetry)
        if not complete and not retry_failed:
            reason = unresolved_attempt(directory, item_telemetry)
            if reason:
                result["unresolved_skipped"][item["session_id"]] = reason
                continue
        if not complete and max_sessions is not None and result["attempted"] >= max_sessions:
            result["deferred"].append(item["session_id"])
            continue
        result["skipped" if complete else "attempted"] += 1
        try:
            success = ingest_session(item, root, item_telemetry, attempts)
        except ProviderAccessError as error:
            result["failed"].append(item["session_id"])
            result["stopped_reason"] = str(error)
            result["deferred"].extend(later["session_id"] for later in items[index+1:]
                                      if not present(root / "sessions" / str(later["year"]) / later["session_id"],
                                                      wants_telemetry(later, telemetry, telemetry_scope)))
            break
        if success:
            if not complete:
                result["succeeded"].append(item["session_id"])
        else:
            result["failed"].append(item["session_id"])
    return result
