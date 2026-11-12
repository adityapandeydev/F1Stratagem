import pytest

from f1data import ratelimit


@pytest.fixture
def clean_limiters():
    found = ratelimit.limiters()
    saved = {info: list(l._timestamps) for info, l in found.items()}
    for limiter in found.values():
        limiter._timestamps.clear()
    yield found
    for info, limiter in found.items():
        limiter._timestamps.clear()
        limiter._timestamps.extend(saved[info])


def test_fastf1_hourly_limiters_are_found(clean_limiters):
    assert "any API: 500 calls/h" in clean_limiters
    assert clean_limiters["any API: 500 calls/h"]._timestamps.maxlen == 500


def test_no_wait_when_budget_is_free(clean_limiters):
    assert ratelimit.seconds_until_budget(now=10_000.0) == 0.0


def test_wait_lasts_until_enough_old_calls_leave_the_window(clean_limiters):
    general = clean_limiters["any API: 500 calls/h"]
    now = 100_000.0
    # 500 calls spread over the last 20 minutes: a burst, like re-checking cached sessions.
    general._timestamps.extend(now - 1200 + i * 2.4 for i in range(500))
    wait = ratelimit.seconds_until_budget(budget=100, now=now)
    oldest_hundredth = now - 1200 + 99 * 2.4
    assert wait == pytest.approx(oldest_hundredth + 3600 - now + ratelimit.SAFETY_SECONDS)
    assert 2400 < wait < 2700


def test_state_survives_a_process_restart(clean_limiters, tmp_path):
    general = clean_limiters["any API: 500 calls/h"]
    now = 200_000.0
    general._timestamps.extend([now - 4000, now - 100, now - 50])
    ratelimit.save(tmp_path, now=now)
    general._timestamps.clear()
    assert ratelimit.restore(tmp_path, now=now + 10) == 2
    assert list(general._timestamps) == [now - 100, now - 50]


def test_missing_or_corrupt_state_is_ignored(clean_limiters, tmp_path):
    assert ratelimit.restore(tmp_path) == 0
    (tmp_path / ratelimit.STATE_FILE).write_text("not json")
    assert ratelimit.restore(tmp_path) == 0
