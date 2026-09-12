"""F1Stratagem Worker — FastF1 data ingestion service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

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


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "f1stratagem-worker",
        "time": datetime.utcnow().isoformat(),
    }
