"""F1Stratagem Worker — FastF1 data ingestion and telemetry service."""

import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingestion.extractor import (
    get_season_schedule,
    extract_session_full,
    compare_telemetry,
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
    session: str  # "FP1", "FP2", "FP3", "Q", "S", "SQ", "R"


class CompareTelemetryRequest(BaseModel):
    year: int
    round_number: int
    session: str
    driver1: str
    lap1: Optional[int] = 0
    driver2: str
    lap2: Optional[int] = 0


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
        logger.info(f"Comparing telemetry: {req.driver1} vs {req.driver2}")
        data = compare_telemetry(
            req.year,
            req.round_number,
            req.session,
            req.driver1,
            req.lap1 or 0,
            req.driver2,
            req.lap2 or 0,
        )
        return {"status": "success", "data": data}
    except Exception as e:
        logger.error(f"Error comparing telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8081, reload=True)
