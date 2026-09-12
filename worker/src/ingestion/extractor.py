"""F1Stratagem FastF1 Extractor — Multi-driver real GPS and telemetry extractor."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import fastf1

logger = logging.getLogger(__name__)

CACHE_DIR = os.getenv("FASTF1_CACHE_DIR", "./data/fastf1_cache")
TELEMETRY_DIR = os.getenv("TELEMETRY_STORAGE_PATH", "./data/telemetry")

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(TELEMETRY_DIR).mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def get_season_schedule(year: int) -> List[Dict[str, Any]]:
    """Retrieve event schedule for a given year."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    events = []
    for _, row in schedule.iterrows():
        round_num = int(row.get("RoundNumber", 0))
        if round_num == 0:
            continue

        event_data = {
            "round_number": round_num,
            "event_name": str(row.get("EventName", "")),
            "official_name": str(row.get("OfficialEventName", "")),
            "country": str(row.get("Country", "")),
            "location": str(row.get("Location", "")),
            "event_format": str(row.get("EventFormat", "conventional")),
            "event_date": str(row.get("EventDate", "")),
            "circuit": {
                "circuit_key": str(row.get("Location", "")).lower().replace(" ", "_"),
                "name": str(row.get("EventName", "")),
                "country": str(row.get("Country", "")),
                "location": str(row.get("Location", "")),
            },
            "sessions": [],
        }

        for s_idx in range(1, 6):
            s_name = row.get(f"Session{s_idx}")
            s_date = row.get(f"Session{s_idx}Date")
            if pd.notna(s_name) and str(s_name).strip():
                s_type = "race" if "race" in str(s_name).lower() else (
                    "qualifying" if "qualifying" in str(s_name).lower() else (
                        "sprint" if "sprint" in str(s_name).lower() else "practice"
                    )
                )
                event_data["sessions"].append({
                    "session_name": str(s_name),
                    "session_type": s_type,
                    "session_date": str(s_date) if pd.notna(s_date) else None,
                })

        events.append(event_data)
    return events


def extract_session_full(year: int, round_num: int, session_identifier: str) -> Dict[str, Any]:
    """Load session, extract laps, results, weather, and write real GPS telemetry to Parquet."""
    logger.info(f"Loading FastF1 session: year={year}, round={round_num}, session={session_identifier}")
    session = fastf1.get_session(year, round_num, session_identifier)
    session.load(laps=True, telemetry=True, weather=True)

    event_info = {
        "event_name": session.event.get("EventName", ""),
        "official_name": session.event.get("OfficialEventName", ""),
        "country": session.event.get("Country", ""),
        "location": session.event.get("Location", ""),
        "round_number": int(session.event.get("RoundNumber", round_num)),
        "event_date": str(session.event.get("EventDate", "")),
    }

    circuit_info = {
        "circuit_key": str(session.event.get("Location", "")).lower().replace(" ", "_"),
        "name": session.event.get("EventName", ""),
        "country": session.event.get("Country", ""),
        "location": session.event.get("Location", ""),
    }

    drivers_list = []
    teams_list = []
    teams_seen = set()

    for drv_num in session.drivers:
        drv_info = session.get_driver(drv_num)
        abbr = str(drv_info.get("Abbreviation", drv_num))
        team_name = str(drv_info.get("TeamName", "Unknown"))
        team_color = str(drv_info.get("TeamColor", "CCCCCC"))
        if not team_color.startswith("#") and team_color:
            team_color = f"#{team_color}"

        drivers_list.append({
            "driver_number": str(drv_info.get("DriverNumber", drv_num)),
            "broadcast_name": str(drv_info.get("BroadcastName", abbr)),
            "abbreviation": abbr,
            "driver_id": str(drv_info.get("DriverId", abbr.lower())),
            "first_name": str(drv_info.get("FirstName", "")),
            "last_name": str(drv_info.get("LastName", "")),
            "full_name": str(drv_info.get("FullName", "")),
            "country_code": str(drv_info.get("CountryCode", "")),
            "headshot_url": str(drv_info.get("HeadshotUrl", "")),
            "team_name": team_name,
            "team_color": team_color,
        })

        if team_name not in teams_seen:
            teams_seen.add(team_name)
            teams_list.append({
                "name": team_name,
                "team_id": team_name.lower().replace(" ", "_"),
                "color": team_color,
            })

    # Results
    results_list = []
    if session.results is not None and not session.results.empty:
        for _, r in session.results.iterrows():
            pos = int(r["Position"]) if pd.notna(r.get("Position")) else None
            grid = int(r["GridPosition"]) if pd.notna(r.get("GridPosition")) else None
            pts = float(r["Points"]) if pd.notna(r.get("Points")) else 0.0

            q1_ms = int(r["Q1"].total_seconds() * 1000) if pd.notna(r.get("Q1")) and hasattr(r["Q1"], "total_seconds") else None
            q2_ms = int(r["Q2"].total_seconds() * 1000) if pd.notna(r.get("Q2")) and hasattr(r["Q2"], "total_seconds") else None
            q3_ms = int(r["Q3"].total_seconds() * 1000) if pd.notna(r.get("Q3")) and hasattr(r["Q3"], "total_seconds") else None
            time_ms = int(r["Time"].total_seconds() * 1000) if pd.notna(r.get("Time")) and hasattr(r["Time"], "total_seconds") else None

            results_list.append({
                "driver_abbreviation": str(r.get("Abbreviation", "")),
                "team_name": str(r.get("TeamName", "")),
                "position": pos,
                "classified_position": str(r.get("ClassifiedPosition", "")),
                "grid_position": grid,
                "q1_time_ms": q1_ms,
                "q2_time_ms": q2_ms,
                "q3_time_ms": q3_ms,
                "race_time_ms": time_ms,
                "status": str(r.get("Status", "Finished")),
                "points": pts,
            })

    # Laps & Telemetry
    laps_list = []
    telemetry_frames = []

    if session.laps is not None and not session.laps.empty:
        for _, lap in session.laps.iterlaps():
            lap_num = int(lap.get("LapNumber", 0))
            drv_abbr = str(lap.get("Driver", ""))

            lap_time_ms = int(lap["LapTime"].total_seconds() * 1000) if pd.notna(lap.get("LapTime")) and hasattr(lap["LapTime"], "total_seconds") else None
            s1_ms = int(lap["Sector1Time"].total_seconds() * 1000) if pd.notna(lap.get("Sector1Time")) and hasattr(lap["Sector1Time"], "total_seconds") else None
            s2_ms = int(lap["Sector2Time"].total_seconds() * 1000) if pd.notna(lap.get("Sector2Time")) and hasattr(lap["Sector2Time"], "total_seconds") else None
            s3_ms = int(lap["Sector3Time"].total_seconds() * 1000) if pd.notna(lap.get("Sector3Time")) and hasattr(lap["Sector3Time"], "total_seconds") else None

            laps_list.append({
                "driver_abbreviation": drv_abbr,
                "lap_number": lap_num,
                "lap_time_ms": lap_time_ms,
                "sector1_ms": s1_ms,
                "sector2_ms": s2_ms,
                "sector3_ms": s3_ms,
                "speed_i1": float(lap["SpeedI1"]) if pd.notna(lap.get("SpeedI1")) else None,
                "speed_i2": float(lap["SpeedI2"]) if pd.notna(lap.get("SpeedI2")) else None,
                "speed_fl": float(lap["SpeedFL"]) if pd.notna(lap.get("SpeedFL")) else None,
                "speed_st": float(lap["SpeedST"]) if pd.notna(lap.get("SpeedST")) else None,
                "compound": str(lap.get("Compound", "")),
                "tyre_life": float(lap["TyreLife"]) if pd.notna(lap.get("TyreLife")) else None,
                "fresh_tyre": bool(lap.get("FreshTyre", False)) if pd.notna(lap.get("FreshTyre")) else None,
                "stint": int(lap["Stint"]) if pd.notna(lap.get("Stint")) else None,
                "is_personal_best": bool(lap.get("IsPersonalBest", False)) if pd.notna(lap.get("IsPersonalBest")) else False,
                "is_deleted": bool(lap.get("Deleted", False)) if pd.notna(lap.get("Deleted")) else False,
                "track_status": str(lap.get("TrackStatus", "")),
                "position": int(lap["Position"]) if pd.notna(lap.get("Position")) else None,
            })

            # Extract car telemetry with REAL GPS X, Y, Z
            if pd.notna(lap.get("LapTime")) and not lap.get("Deleted", False):
                try:
                    tel = lap.get_telemetry()
                    if tel is not None and not tel.empty:
                        df_tel = pd.DataFrame({
                            "driver": drv_abbr,
                            "lap": lap_num,
                            "distance": tel["Distance"].round(2),
                            "speed": tel["Speed"].round(1),
                            "throttle": tel["Throttle"].round(1),
                            "brake": tel["Brake"].astype(int),
                            "gear": tel["nGear"].astype(int),
                            "rpm": tel["RPM"].round(0),
                            "drs": tel["DRS"].astype(int),
                            "x": tel["X"].round(1) if "X" in tel.columns else 0.0,
                            "y": tel["Y"].round(1) if "Y" in tel.columns else 0.0,
                            "z": tel["Z"].round(1) if "Z" in tel.columns else 0.0,
                        })
                        telemetry_frames.append(df_tel)
                except Exception as e:
                    logger.debug(f"Telemetry extract skipped for {drv_abbr} lap {lap_num}: {e}")

    # Weather
    weather_list = []
    if session.weather_data is not None and not session.weather_data.empty:
        for _, w in session.weather_data.iterrows():
            offset_ms = int(w["Time"].total_seconds() * 1000) if pd.notna(w.get("Time")) and hasattr(w["Time"], "total_seconds") else 0
            weather_list.append({
                "time_offset_ms": offset_ms,
                "air_temp": float(w["AirTemp"]) if pd.notna(w.get("AirTemp")) else None,
                "track_temp": float(w["TrackTemp"]) if pd.notna(w.get("TrackTemp")) else None,
                "humidity": float(w["Humidity"]) if pd.notna(w.get("Humidity")) else None,
                "pressure": float(w["Pressure"]) if pd.notna(w.get("Pressure")) else None,
                "wind_speed": float(w["WindSpeed"]) if pd.notna(w.get("WindSpeed")) else None,
                "wind_direction": int(w["WindDirection"]) if pd.notna(w.get("WindDirection")) else None,
                "rainfall": bool(w.get("Rainfall", False)),
            })

    parquet_path = f"{TELEMETRY_DIR}/{year}_{round_num}_{session_identifier.lower()}.parquet"
    laps_json_path = f"{TELEMETRY_DIR}/{year}_{round_num}_{session_identifier.lower()}_laps.json"
    total_tel_points = 0
    if telemetry_frames:
        combined_tel = pd.concat(telemetry_frames, ignore_index=True)
        combined_tel.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        total_tel_points = len(combined_tel)
        logger.info(f"Saved {total_tel_points} telemetry points to {parquet_path}")

    # Also persist lightweight laps metadata for instant context-aware lap selection
    laps_by_driver: Dict[str, List[Dict[str, Any]]] = {}
    for l in laps_list:
        d_abbr = l.get("driver_abbreviation", "")
        if not d_abbr:
            continue
        if d_abbr not in laps_by_driver:
            laps_by_driver[d_abbr] = []
        if l.get("lap_time_ms") and not l.get("is_deleted", False):
            laps_by_driver[d_abbr].append({
                "lap": l["lap_number"],
                "time_ms": l["lap_time_ms"],
                "compound": l.get("compound", "UNKNOWN"),
                "pb": l.get("is_personal_best", False),
            })
    if laps_by_driver:
        try:
            with open(laps_json_path, "w", encoding="utf-8") as f:
                json.dump(laps_by_driver, f, indent=2)
            logger.info(f"Saved laps metadata to {laps_json_path}")
        except Exception as e:
            logger.warning(f"Failed to write laps metadata JSON: {e}")

    return {
        "event": event_info,
        "circuit": circuit_info,
        "drivers": drivers_list,
        "teams": teams_list,
        "results": results_list,
        "laps": laps_list,
        "weather": weather_list,
        "telemetry_file": parquet_path if total_tel_points > 0 else None,
        "telemetry_row_count": total_tel_points,
    }


DRIVER_TEAMS_2024: Dict[str, Dict[str, str]] = {
    "VER": {"full_name": "Max Verstappen", "team_name": "Red Bull Racing", "color": "#3671C6"},
    "PER": {"full_name": "Sergio Perez", "team_name": "Red Bull Racing", "color": "#3671C6"},
    "LEC": {"full_name": "Charles Leclerc", "team_name": "Ferrari", "color": "#E8002D"},
    "SAI": {"full_name": "Carlos Sainz", "team_name": "Ferrari", "color": "#E8002D"},
    "RUS": {"full_name": "George Russell", "team_name": "Mercedes", "color": "#27F4D2"},
    "HAM": {"full_name": "Lewis Hamilton", "team_name": "Mercedes", "color": "#27F4D2"},
    "NOR": {"full_name": "Lando Norris", "team_name": "McLaren", "color": "#FF8000"},
    "PIA": {"full_name": "Oscar Piastri", "team_name": "McLaren", "color": "#FF8000"},
    "ALO": {"full_name": "Fernando Alonso", "team_name": "Aston Martin", "color": "#229971"},
    "STR": {"full_name": "Lance Stroll", "team_name": "Aston Martin", "color": "#229971"},
    "TSU": {"full_name": "Yuki Tsunoda", "team_name": "RB", "color": "#6692FF"},
    "RIC": {"full_name": "Daniel Ricciardo", "team_name": "RB", "color": "#6692FF"},
    "ALB": {"full_name": "Alexander Albon", "team_name": "Williams", "color": "#64C4FF"},
    "SAR": {"full_name": "Logan Sargeant", "team_name": "Williams", "color": "#64C4FF"},
    "HUL": {"full_name": "Nico Hulkenberg", "team_name": "Haas", "color": "#B6BABD"},
    "MAG": {"full_name": "Kevin Magnussen", "team_name": "Haas", "color": "#B6BABD"},
    "BOT": {"full_name": "Valtteri Bottas", "team_name": "Kick Sauber", "color": "#52E252"},
    "ZHO": {"full_name": "Zhou Guanyu", "team_name": "Kick Sauber", "color": "#52E252"},
    "OCO": {"full_name": "Esteban Ocon", "team_name": "Alpine", "color": "#0093CC"},
    "GAS": {"full_name": "Pierre Gasly", "team_name": "Alpine", "color": "#0093CC"},
}


def compute_running_delta(pts1: List[Dict[str, Any]], pts2: List[Dict[str, Any]]) -> List[float]:
    """Computes physics-based running delta time (seconds) relative to driver 1."""
    if not pts1 or not pts2:
        return []
    try:
        d1 = np.array([p["d"] for p in pts1], dtype=np.float64)
        s1 = np.maximum(np.array([p["spd"] for p in pts1], dtype=np.float64) / 3.6, 2.0)

        d2 = np.array([p["d"] for p in pts2], dtype=np.float64)
        s2 = np.maximum(np.array([p["spd"] for p in pts2], dtype=np.float64) / 3.6, 2.0)

        # Interpolate s2 onto d1 distance grid
        s2_interp = np.interp(d1, d2, s2)

        # dt = (1/v2 - 1/v1) * delta_d
        delta_d = np.diff(d1, prepend=0.0)
        dt_elements = delta_d * (1.0 / s2_interp - 1.0 / s1)
        cum_delta = np.cumsum(dt_elements)
        return [round(float(x), 4) for x in cum_delta]
    except Exception as e:
        logger.warning(f"Error computing running delta: {e}")
        return []


def format_lap_label(lap_num: int, time_ms: Optional[int], is_pb: bool = False) -> str:
    """Format lap label for context-aware dropdown."""
    if time_ms is None or time_ms <= 0:
        return f"Lap {lap_num}"
    mins = time_ms // 60000
    secs = (time_ms % 60000) / 1000.0
    time_str = f"{mins}:{secs:06.3f}"
    if is_pb:
        return f"Lap {lap_num} • {time_str} (Best)"
    return f"Lap {lap_num} • {time_str}"


def compare_multi_telemetry(year: int, round_num: int, session_identifier: str, driver_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare telemetry for up to 4 drivers with real GPS coordinates and distance grid.
    
    Checks direct Parquet storage first for blazing fast (<60ms) access.
    """
    logger.info(f"Comparing multi-driver telemetry for {len(driver_requests)} drivers in {year} round {round_num} {session_identifier}")

    # Standardize session identifier for file lookup
    s_raw = session_identifier.strip().lower()
    candidate_codes = [s_raw]
    if "qual" in s_raw or s_raw == "q":
        candidate_codes.extend(["q", "qualifying"])
    elif "race" in s_raw or s_raw == "r":
        candidate_codes.extend(["r", "race"])
    elif "fp1" in s_raw or "practice 1" in s_raw:
        candidate_codes.extend(["fp1", "practice 1"])
    elif "fp2" in s_raw or "practice 2" in s_raw:
        candidate_codes.extend(["fp2", "practice 2"])
    elif "fp3" in s_raw or "practice 3" in s_raw:
        candidate_codes.extend(["fp3", "practice 3"])

    # Locate parquet file on disk
    target_parquet: Optional[str] = None
    target_laps_json: Optional[str] = None
    for code in candidate_codes:
        p_path = os.path.join(TELEMETRY_DIR, f"{year}_{round_num}_{code}.parquet")
        if os.path.exists(p_path):
            target_parquet = p_path
            l_path = os.path.join(TELEMETRY_DIR, f"{year}_{round_num}_{code}_laps.json")
            if os.path.exists(l_path):
                target_laps_json = l_path
            break

    # If Parquet exists on disk, read directly without invoking heavy FastF1 session load!
    if target_parquet and os.path.exists(target_parquet):
        logger.info(f"Reading telemetry directly from Parquet dataset: {target_parquet}")
        df = pd.read_parquet(target_parquet)

        laps_meta_map: Dict[str, List[Dict[str, Any]]] = {}
        if target_laps_json and os.path.exists(target_laps_json):
            try:
                with open(target_laps_json, "r", encoding="utf-8") as f:
                    laps_meta_map = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load laps JSON {target_laps_json}: {e}")

        drivers_output = []
        telemetry_streams = []

        for d_req in driver_requests[:4]:
            abbr = d_req.get("driver", "").upper()
            req_lap = int(d_req.get("lap", 0))

            drv_df = df[df["driver"] == abbr]
            if drv_df.empty:
                logger.warning(f"Driver {abbr} not found in Parquet dataset")
                continue

            # Build available laps list for context-aware selection
            drv_laps_meta = laps_meta_map.get(abbr, [])
            available_laps_list = []

            # If we have laps metadata, use real times and personal best flags
            if drv_laps_meta:
                # Sort by lap number
                sorted_meta = sorted(drv_laps_meta, key=lambda x: x["lap"])
                # Identify overall fastest lap
                best_lap_meta = min(sorted_meta, key=lambda x: x.get("time_ms", float("inf")))
                for lm in sorted_meta:
                    l_num = lm["lap"]
                    # Ensure lap exists in parquet
                    if not drv_df[drv_df["lap"] == l_num].empty:
                        t_ms = lm.get("time_ms")
                        is_best = (l_num == best_lap_meta.get("lap"))
                        available_laps_list.append({
                            "lap": l_num,
                            "time_ms": t_ms,
                            "label": format_lap_label(l_num, t_ms, is_best),
                            "is_pb": is_best,
                        })
                chosen_lap_num = req_lap if req_lap > 0 else best_lap_meta.get("lap", sorted_meta[0]["lap"])
                chosen_lap_time = next((l["time_ms"] for l in available_laps_list if l["lap"] == chosen_lap_num), None)
            else:
                # Fallback: extract distinct laps directly from parquet
                unique_laps = sorted(drv_df["lap"].unique().tolist())
                for l_num in unique_laps:
                    available_laps_list.append({
                        "lap": int(l_num),
                        "time_ms": None,
                        "label": f"Lap {l_num}",
                        "is_pb": False,
                    })
                chosen_lap_num = req_lap if req_lap > 0 else unique_laps[0]
                chosen_lap_time = None

            # Slice telemetry for chosen lap
            lap_df = drv_df[drv_df["lap"] == chosen_lap_num]
            if lap_df.empty:
                # If requested lap was not found, fallback to first available
                chosen_lap_num = available_laps_list[0]["lap"]
                lap_df = drv_df[drv_df["lap"] == chosen_lap_num]

            # Driver metadata from authoritative 2024 map
            d_info = DRIVER_TEAMS_2024.get(abbr, {
                "full_name": abbr,
                "team_name": "Formula 1",
                "color": "#ffffff",
            })

            driver_meta = {
                "abbreviation": abbr,
                "full_name": d_info["full_name"],
                "team_name": d_info["team_name"],
                "color": d_info["color"],
                "lap_number": int(chosen_lap_num),
                "lap_time_ms": chosen_lap_time,
                "available_laps": available_laps_list,
            }
            drivers_output.append(driver_meta)

            # Build telemetry points stream
            pts = []
            for _, row in lap_df.iterrows():
                pts.append({
                    "d": round(float(row["distance"]), 2),
                    "spd": round(float(row["speed"]), 1),
                    "thr": round(float(row["throttle"]), 1),
                    "brk": int(row["brake"]),
                    "rpm": round(float(row["rpm"]), 0),
                    "gear": int(row["gear"]),
                    "drs": int(row["drs"]),
                    "x": round(float(row["x"]), 1) if "x" in row else 0.0,
                    "y": round(float(row["y"]), 1) if "y" in row else 0.0,
                    "z": round(float(row["z"]), 1) if "z" in row else 0.0,
                })
            telemetry_streams.append(pts)

        # Compute running physics delta relative to driver 1
        deltas: List[float] = []
        if len(telemetry_streams) >= 2:
            deltas = compute_running_delta(telemetry_streams[0], telemetry_streams[1])

        return {
            "drivers": drivers_output,
            "telemetry": telemetry_streams,
            "time_delta": deltas,
        }

    # Fallback to FastF1 session load if Parquet not yet generated
    logger.info(f"Parquet dataset not found on disk, falling back to FastF1 session load: {session_identifier}")
    session = fastf1.get_session(year, round_num, session_identifier)
    session.load(laps=True, telemetry=True, weather=False)

    drivers_output = []
    telemetry_streams = []
    ref_lap_obj = None

    for d_req in driver_requests[:4]:
        abbr = d_req.get("driver")
        req_lap = d_req.get("lap", 0)

        drv_laps = session.laps.pick_driver(abbr)
        if drv_laps.empty:
            continue

        lap_obj = drv_laps[drv_laps["LapNumber"] == req_lap].iloc[0] if req_lap > 0 else drv_laps.pick_fastest()
        if lap_obj is None or pd.isna(lap_obj.get("LapTime")):
            continue

        if ref_lap_obj is None:
            ref_lap_obj = lap_obj

        tel = lap_obj.get_telemetry()
        lap_time_ms = int(lap_obj["LapTime"].total_seconds() * 1000)

        drv_info = session.get_driver(abbr)
        team_color = str(drv_info.get("TeamColor", "3b82f6"))
        if not team_color.startswith("#"):
            team_color = f"#{team_color}"

        # Collect available valid laps for this driver
        avail_laps = []
        for _, l in drv_laps.iterlaps():
            if pd.notna(l.get("LapTime")) and not l.get("Deleted", False):
                t_ms = int(l["LapTime"].total_seconds() * 1000)
                is_pb = bool(l.get("IsPersonalBest", False))
                l_num = int(l["LapNumber"])
                avail_laps.append({
                    "lap": l_num,
                    "time_ms": t_ms,
                    "label": format_lap_label(l_num, t_ms, is_pb),
                    "is_pb": is_pb,
                })

        driver_meta = {
            "abbreviation": abbr,
            "full_name": str(drv_info.get("FullName", abbr)),
            "team_name": str(drv_info.get("TeamName", "")),
            "color": team_color,
            "lap_number": int(lap_obj["LapNumber"]),
            "lap_time_ms": lap_time_ms,
            "available_laps": avail_laps,
        }
        drivers_output.append(driver_meta)

        pts = []
        for _, row in tel.iterrows():
            pts.append({
                "d": round(float(row.get("Distance", 0.0)), 2),
                "spd": round(float(row.get("Speed", 0.0)), 1),
                "thr": round(float(row.get("Throttle", 0.0)), 1),
                "brk": int(row.get("Brake", 0)),
                "rpm": round(float(row.get("RPM", 0.0)), 0),
                "gear": int(row.get("nGear", 0)),
                "drs": int(row.get("DRS", 0)),
                "x": round(float(row.get("X", 0.0)), 1) if "X" in row else 0.0,
                "y": round(float(row.get("Y", 0.0)), 1) if "Y" in row else 0.0,
                "z": round(float(row.get("Z", 0.0)), 1) if "Z" in row else 0.0,
            })
        telemetry_streams.append(pts)

    deltas = []
    if len(drivers_output) >= 2 and ref_lap_obj is not None:
        try:
            d2_lap_obj = session.laps.pick_driver(drivers_output[1]["abbreviation"]).pick_fastest()
            delta_time, _, _ = fastf1.utils.delta_time(ref_lap_obj, d2_lap_obj)
            deltas = [round(float(d), 4) for d in delta_time]
        except Exception as e:
            logger.warning(f"Failed to calculate FastF1 delta time: {e}")

    return {
        "drivers": drivers_output,
        "telemetry": telemetry_streams,
        "time_delta": deltas,
    }
