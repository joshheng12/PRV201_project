"""Local-only API for metadata analysis. Interactive docs at /docs."""
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
from .pipeline import run

app = FastAPI(title="VideoLens PRV-201", version="0.1.0",
              description="Local prototype. POST metadata; receive computed statistics. No cloud calls or arbitrary file paths.")
TRAINING = Path(__file__).resolve().parents[1] / "data" / "seed_training.csv"

class AnalysisRequest(BaseModel):
    videos: list[dict] = Field(min_length=10, max_length=5000)

@app.get("/health")
def health():
    return {"status": "ok", "platform": "youtube", "cloud_calls": False}

@app.post("/analyze")
def analyze(request: AnalysisRequest):
    try:
        with tempfile.TemporaryDirectory(prefix="videolens-") as directory:
            root = Path(directory)
            pd.DataFrame(request.videos).to_csv(root / "input.csv", index=False)
            return run(root / "input.csv", root / "results", TRAINING)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
