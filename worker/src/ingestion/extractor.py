"""F1Stratagem FastF1 Extractor — Multi-driver real GPS and telemetry extractor."""

import os
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
    total_tel_points = 0
    if telemetry_frames:
        combined_tel = pd.concat(telemetry_frames, ignore_index=True)
        combined_tel.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        total_tel_points = len(combined_tel)
        logger.info(f"Saved {total_tel_points} telemetry points to {parquet_path}")

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


def compare_multi_telemetry(year: int, round_num: int, session_identifier: str, driver_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare telemetry for up to 4 drivers with real GPS coordinates and distance grid."""
    logger.info(f"Comparing multi-driver telemetry for {len(driver_requests)} drivers in {year} round {round_num} {session_identifier}")
    session = fastf1.get_session(year, round_num, session_identifier)
    session.load(laps=True, telemetry=True, weather=False)

    drivers_output = []
    telemetry_streams = []

    # Pick reference driver for distance grid
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

        driver_meta = {
            "abbreviation": abbr,
            "full_name": str(drv_info.get("FullName", abbr)),
            "team_name": str(drv_info.get("TeamName", "")),
            "color": team_color,
            "lap_number": int(lap_obj["LapNumber"]),
            "lap_time_ms": lap_time_ms,
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

    # Compute deltas relative to driver 1
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
