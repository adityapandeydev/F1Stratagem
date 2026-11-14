import numpy as np
import pandas as pd
import pytest

from predictor import track
from predictor.archive import Archive, CutoffViolation

START = pd.Timestamp("2023-07-01 14:00")


def synthetic_lap(segments, car_hz=4.0, pos_hz=3.7, lead_m=300.0, jitter=0.03, seed=0):
    """(car, pos, lap start, lap end) for a lap built from segments, with `lead_m` of
    straight before and after it. Segments: ("straight", length m, km/h) or ("arc",
    radius m, degrees, "L"/"R", km/h). Car and position samples have different rates
    and jittered timestamps."""
    points, x, y, heading = [], 0.0, 0.0, 0.0
    lead = [("straight", lead_m, segments[0][-1])]
    for segment in lead + list(segments) + [("straight", lead_m, segments[-1][-1])]:
        if segment[0] == "straight":
            steps, turn, speed = int(segment[1]), 0.0, segment[2]
        else:
            _, radius, degrees, direction, speed = segment
            steps, turn = int(np.radians(degrees) * radius), (1.0 if direction == "L" else -1.0) / radius
        for _ in range(steps):
            heading += turn
            x, y = x + np.cos(heading), y + np.sin(heading)
            points.append((x, y, speed))
    xs, ys, speed = (np.array(c, dtype=float) for c in zip(*points))
    target = speed.copy()
    for i in range(len(speed) - 2, -1, -1):          # brake into slower parts
        speed[i] = min(speed[i], speed[i + 1] + 2.5)
    for i in range(1, len(speed)):                    # accelerate out of them
        speed[i] = min(speed[i], speed[i - 1] + 1.0)
    seconds = np.concatenate([[0.0], np.cumsum(1.0 / (speed[1:] / 3.6))])
    rng = np.random.default_rng(seed)

    def samples(rate):
        t = np.arange(0.0, seconds[-1], 1.0 / rate)
        return np.sort(np.clip(t + rng.uniform(-jitter, jitter, len(t)), 0, seconds[-1]))

    tc, tp = samples(car_hz), samples(pos_hz)
    v = np.interp(tc, seconds, speed)
    lifting = v < np.interp(tc, seconds, target) - 1
    car = pd.DataFrame({"Date": START + pd.to_timedelta(tc, unit="s"), "Speed": v,
                        "Throttle": np.where(lifting, 0.0, 100.0), "Brake": lifting})
    pos = pd.DataFrame({"Date": START + pd.to_timedelta(tp, unit="s"),
                        "X": np.interp(tp, seconds, xs) * 10, "Y": np.interp(tp, seconds, ys) * 10})
    index = np.arange(len(seconds))
    start = START + pd.to_timedelta(np.interp(lead_m, index, seconds), unit="s")
    end = START + pd.to_timedelta(np.interp(len(seconds) - 1 - lead_m, index, seconds), unit="s")
    return car, pos, start, end


LAP = [("straight", 900, 320), ("arc", 15, 180, "L", 60),            # hairpin after a long straight
       ("straight", 500, 300), ("arc", 250, 70, "R", 300),           # fast, taken flat
       ("straight", 400, 300), ("arc", 120, 90, "R", 150),           # medium
       ("straight", 400, 290), ("arc", 40, 50, "L", 95),             # chicane: left then right
       ("straight", 20, 95), ("arc", 40, 50, "R", 95),
       ("straight", 500, 280), ("arc", 45, 100, "L", 105),           # slow
       ("straight", 600, 300)]


def trace_of(segments, **kwargs):
    return track.lap_trace(*synthetic_lap(segments, **kwargs))


def test_corners_are_found_in_order_and_classified():
    table = track.corners(trace_of(LAP))
    assert list(table.direction) == ["L", "R", "R", "L", "R", "L"]
    assert list(table.hairpin.round()) == [1, 0, 0, 0, 0, 0]
    assert list(table.chicane.round()) == [0, 0, 0, 1, 1, 0]
    assert table.turn_deg.iloc[0] == pytest.approx(180, abs=20)
    assert table.min_speed.iloc[0] == pytest.approx(60, abs=8)


def test_an_s_bend_is_split_where_the_turn_changes_direction():
    s_bend = [("straight", 600, 300), ("arc", 60, 80, "L", 150), ("arc", 60, 80, "R", 150), ("straight", 600, 300)]
    table = track.corners(trace_of(s_bend))
    assert list(table.direction) == ["L", "R"]
    assert all(table.turn_deg > 60)


def test_lap_edges_add_no_corners():
    assert track.corners(trace_of([("straight", 2000, 300)])).empty


def test_shares_sum_to_one_and_bands_are_soft():
    p = track.profile(trace_of(LAP))
    assert sum(p[f"share_{b}"] for b in ("straight", "slow", "medium", "fast")) == pytest.approx(1.0)
    assert p["share_fast"] > 0 and p["share_slow"] > 0 and p["share_medium"] > 0
    assert p["hairpins_per_km"] > 0 and p["chicanes_per_km"] > 0 and p["heavy_braking_per_km"] > 0
    assert 0.3 < p["share_full_throttle"] < 1
    at_edge, below, above = (track.band_weights(v) for v in (120.0, 115.0, 125.0))
    assert at_edge["slow"] == pytest.approx(0.5) and at_edge["medium"] == pytest.approx(0.5, abs=1e-3)
    assert 0.5 < below["slow"] < 0.8 and 0.5 < above["medium"] < 0.8     # no cliff at the boundary
    for v in (40.0, 160.0, 300.0):
        assert sum(track.band_weights(v).values()) == pytest.approx(1.0)


def test_one_braking_zone_counts_once():
    complex_ = [("straight", 900, 300), ("arc", 40, 90, "R", 110), ("straight", 30, 120),
                ("arc", 40, 90, "L", 110), ("straight", 900, 300)]
    assert track.braking_zones(trace_of(complex_)) == pytest.approx(1, abs=0.05)


def test_a_lap_with_a_telemetry_gap_is_not_used():
    car, pos, start, end = synthetic_lap(LAP)
    middle = start + (end - start) / 2
    gap = pos[(pos.Date < middle) | (pos.Date > middle + pd.Timedelta(seconds=3))]
    assert track.lap_trace(car, gap, start, end) is None
    assert track.lap_trace(car, pos, start, end) is not None


def test_the_reference_lap_is_the_median_of_laps():
    reference = track.reference_lap([trace_of(LAP, seed=s) for s in range(5)])
    table = track.corners(reference)
    assert len(table) == 6 and table.hairpin.sum() == pytest.approx(1, abs=0.05)


def laps_table(compounds, statuses):
    n = len(compounds)
    return pd.DataFrame({"DriverNumber": [str(i) for i in range(n)], "LapTime": pd.to_timedelta(np.arange(n) + 80, unit="s"),
                         "LapStartDate": [START] * n, "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "IsAccurate": True,
                         "Deleted": False, "Compound": compounds, "TrackStatus": statuses})


def test_clean_laps_drop_wet_tyres_and_yellow_flags():
    laps = laps_table(["SOFT", "INTERMEDIATE", "MEDIUM", "WET", "SOFT", None], ["1", "1", "12", "1", "1", "1"])
    assert list(track.clean_laps(laps).DriverNumber) == ["0", "4"]


def test_telemetry_respects_the_cutoff(tmp_path):
    from test_pace import write_circuit_archive
    write_circuit_archive(tmp_path)
    archive = Archive(tmp_path)
    with pytest.raises(CutoffViolation):
        archive.view(pd.Timestamp("2021-07-02 11:30", tz="UTC")).telemetry("2021-01-2", "0")
    later = archive.view(pd.Timestamp("2023-01-01", tz="UTC"))
    assert later.telemetry("2021-01-2", "0") is None                     # no telemetry in the fixture
    with pytest.raises(ValueError):
        later.telemetry("2021-01-2", "0", kind="weather")


def test_layout_profile_is_the_median_of_sessions():
    profiles = [dict.fromkeys(track.FEATURES, 5.0), None] + [dict.fromkeys(track.FEATURES, v) for v in (0.1, 0.2, 0.9)]
    # only the latest three usable sessions count (the old 5.0 does not)
    median = track.layout_profile(profiles)
    assert median["share_slow"] == pytest.approx(0.25)                # 0.2, renormalised over four shares
    assert sum(median[f"share_{b}"] for b in ("straight", "slow", "medium", "fast")) == pytest.approx(1.0)
    assert median["hairpins_per_km"] == pytest.approx(0.2)
    assert track.layout_profile([None]) is None


def test_every_event_from_2018_to_2024_has_one_layout():
    """Integration test: needs the local data archive; skipped without it."""
    from f1data.paths import default_data_root
    root = default_data_root()
    if not all((root / "reference" / f"{name}.parquet").exists() for name in ("events", "sessions", "entrants")):
        pytest.skip("no local archive")
    archive = Archive(root)
    events = archive.events[(archive.events.season >= 2018) & (archive.events.season <= 2024) & ~archive.events.is_testing]
    table = track.layouts()
    missing = [e.event_key for e in events.itertuples() if track.layout_of(e.circuit_id, e.season, e.event_key, table) is None]
    assert missing == []
    assert track.layout_of("bahrain", 2020, "2020-15", table) == "bahrain/grand_prix"
    assert track.layout_of("bahrain", 2020, "2020-16", table) == "bahrain/outer"
    assert track.layout_of("catalunya", 2023, "2023-07", table) == "catalunya/no_chicane"


def test_a_joined_slow_and_fast_corner_keeps_its_fast_part_fast():
    """Two same-direction pieces close together are one corner, but the fast piece's
    time stays in the fast band."""
    joined = [("straight", 900, 300), ("arc", 40, 90, "R", 100), ("straight", 35, 210),
              ("arc", 200, 60, "R", 230), ("straight", 900, 300)]
    trace = trace_of(joined)
    table = track.corners(trace)
    assert len(table) == 1 and len(table.pieces.iloc[0]) == 2
    p = track.profile(trace, table)
    slow_piece, second_piece = table.pieces.iloc[0]
    assert second_piece[3] > slow_piece[3] + 30           # each piece keeps its own minimum speed
    assert p["share_medium"] + p["share_fast"] > 0.3 * p["share_slow"]   # not all banded by the complex's minimum


def test_esses_give_one_chicane_and_a_slow_short_turn_is_not_a_hairpin():
    esses = [("straight", 900, 300), ("arc", 40, 50, "L", 95), ("straight", 20, 95), ("arc", 40, 50, "R", 95),
             ("straight", 20, 95), ("arc", 40, 50, "L", 95), ("straight", 900, 300)]
    table = track.corners(trace_of(esses))
    assert len(table) == 3 and table.chicane.round().sum() == 2       # one pair; the third corner unpaired
    assert table.hairpin.max() < 0.1                                  # slow but only 50 degrees each


def test_a_brief_uptick_does_not_split_a_braking_zone():
    trace = trace_of([("straight", 1500, 300)])
    speed = trace.speed.to_numpy().copy()
    profile_ = np.concatenate([np.linspace(300, 200, 60), [203.0, 201.0], np.linspace(200, 90, 60)])
    speed[100:100 + len(profile_)] = profile_
    speed[100 + len(profile_):] = 90
    assert track.braking_zones(trace.assign(speed=speed)) == pytest.approx(1, abs=0.01)


def test_a_session_with_wet_running_has_no_profile():
    laps = laps_table(["SOFT", "INTERMEDIATE", "SOFT"], ["1", "1", "1"])
    assert track.wet_running(laps) and not track.wet_running(laps_table(["SOFT", "MEDIUM"], ["1", "1"]))

    class View:
        def laps(self, session_id):
            return laps

    assert track.session_profile(View(), "2022-17-4") is None


def test_band_times_split_the_lap_and_a_lap_matches_itself():
    traces = [trace_of(LAP, seed=s) for s in range(3)]
    reference = track.reference_lap(traces)
    table = track.corners(reference)
    times = track.band_times(traces[0], reference, table)
    assert set(times) == set(track.BANDS)
    assert sum(times.values()) == pytest.approx(traces[0].time.iloc[-1] - traces[0].time.iloc[0], rel=0.01)
    again = track.band_times(traces[0], reference, table)
    assert all(times[b] == again[b] for b in track.BANDS)
    slower = traces[0].assign(time=traces[0].time * 1.01)            # 1% slower everywhere
    assert all(track.band_times(slower, reference, table)[b] > times[b] for b in track.BANDS if times[b] > 0)


def test_layout_lookup_without_the_archive():
    """Portable: the layout table alone, no archive needed."""
    table = track.layouts()
    assert track.layout_of("bahrain", 2020, "2020-15", table) == "bahrain/grand_prix"
    assert track.layout_of("bahrain", 2020, "2020-16", table) == "bahrain/outer"
    assert track.layout_of("catalunya", 2021, "2021-04", table) == "catalunya/with_chicane_2021"
    assert track.layout_of("catalunya", 2023, "2023-07", table) == "catalunya/no_chicane"
    assert track.layout_of("nowhere", 2021, "2021-01", table) is None
    assert (table.layout_from <= table.layout_to).all() and table.turns.gt(0).all()
