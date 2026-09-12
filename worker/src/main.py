"""F1Stratagem Worker - FastF1 data ingestion and telemetry service."""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.extractor import (
    get_season_schedule,
    extract_session_full,
    compare_multi_telemetry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="F1Stratagem Worker",
    description="FastF1 data ingestion and telemetry extraction service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractSessionRequest(BaseModel):
    year: int
    round_number: int
    session: str


class DriverLapRequest(BaseModel):
    driver: str
    lap: Optional[int] = 0


class CompareTelemetryRequest(BaseModel):
    year: int
    round_number: int
    session: str
    driver1: Optional[str] = None
    lap1: Optional[int] = 0
    driver2: Optional[str] = None
    lap2: Optional[int] = 0
    drivers: Optional[List[DriverLapRequest]] = None


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "f1stratagem-worker",
        "time": datetime.utcnow().isoformat(),
    }


@app.get("/schedule/{year}")
async def schedule(year: int):
    try:
        events = get_season_schedule(year)
        return {"year": year, "events": events}
    except Exception as e:
        logger.error(f"Error fetching schedule for {year}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/extract")
async def extract_session(req: ExtractSessionRequest):
    try:
        logger.info(f"Extracting session: year={req.year}, round={req.round_number}, session={req.session}")
        data = extract_session_full(req.year, req.round_number, req.session)
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Error extracting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/telemetry/compare")
async def compare(req: CompareTelemetryRequest):
    try:
        driver_reqs: List[Dict[str, Any]] = []
        if req.drivers and len(req.drivers) > 0:
            driver_reqs = [{"driver": d.driver, "lap": d.lap or 0} for d in req.drivers]
        else:
            if req.driver1:
                driver_reqs.append({"driver": req.driver1, "lap": req.lap1 or 0})
            if req.driver2:
                driver_reqs.append({"driver": req.driver2, "lap": req.lap2 or 0})

        logger.info(f"Comparing telemetry for {len(driver_reqs)} drivers in {req.year} round {req.round_number} {req.session}")
        data = compare_multi_telemetry(
            req.year,
            req.round_number,
            req.session,
            driver_reqs,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Error comparing telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8081, reload=True)
