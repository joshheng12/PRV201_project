import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from videolens.data import make_demo, load, validate
from videolens.models import classify, relations, compare_models
from videolens.pipeline import run, evaluate
from videolens.api import app
from videolens.youtube import duration_seconds

@pytest.fixture
def fixture_data(tmp_path):
    make_demo(tmp_path / "data")
    return tmp_path / "data"

@pytest.mark.parametrize("column,value", [("views", 0), ("likes", -1), ("comments", ""),
    ("duration_seconds", float("inf")), ("views", 1.5), ("data_origin", "fake"), ("platform", "tiktok")])
def test_reject_invalid_metrics(fixture_data, column, value):
    frame = pd.read_csv(fixture_data / "demo_videos.csv", keep_default_na=False)
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate(frame)

def test_duplicate_ids_rejected(fixture_data):
    frame = pd.read_csv(fixture_data / "demo_videos.csv", keep_default_na=False)
    frame.loc[1, "video_id"] = frame.loc[0, "video_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate(frame)

def test_missing_columns(fixture_data):
    frame = pd.read_csv(fixture_data / "demo_videos.csv").drop(columns="views")
    with pytest.raises(ValueError, match="Missing columns"):
        validate(frame)

def test_group_holdout_no_creator_overlap(fixture_data):
    df = load(fixture_data / "demo_videos.csv")
    metrics = compare_models(df)
    indexed = df.set_index("video_id")
    a = set(indexed.loc[metrics["train_video_ids"]].creator_id)
    b = set(indexed.loc[metrics["test_video_ids"]].creator_id)
    assert a.isdisjoint(b)
    assert set(metrics["features"]).isdisjoint({"views", "likes", "comments", "engagement_rate"})
    assert metrics["mean_baseline"]["mae"] >= 0

def test_constant_features_explicit(fixture_data):
    df = load(fixture_data / "demo_videos.csv")
    df["duration_seconds"] = 1
    assert relations(df)[0]["spearman_rho"] is None

def test_offline_end_to_end_and_provenance(fixture_data, tmp_path):
    out = tmp_path / "run"
    result = run(fixture_data / "demo_videos.csv", out, fixture_data / "seed_training.csv")
    assert result["video_count"] == 60
    assert result["contains_synthetic_data"]
    assert result["human_review"]["agreement"] is None
    assert len(result["feature_relationships"]) == 3
    assert result["narratives"] == []
    report = (out / "report.html").read_text(encoding="utf-8")
    assert "SYNTHETIC DEMONSTRATION" in report
    assert report.count("data:image/png;base64,") == 3
    manifest = json.loads((out / "manifest.json").read_text())
    assert not manifest["submission_ready"]
    assert len(manifest["input_sha256"]) == 2
    assert len(pd.read_csv(out / "human_review_template.csv")) == 10
    assert "category" not in pd.read_csv(out / "human_review_template.csv").columns
    assert json.loads((out / "execution_trace.jsonl").read_text().splitlines()[-1])["stage"] == "completed"
    with pytest.raises(ValueError, match="empty"):
        run(fixture_data / "demo_videos.csv", out, fixture_data / "seed_training.csv")

def test_review_cannot_claim_synthetic_evidence(fixture_data, tmp_path):
    df = classify(load(fixture_data / "demo_videos.csv"), fixture_data / "seed_training.csv")
    reviews = df.head(10)[["video_id", "category"]].rename(columns={"category": "human_category"})
    reviews["reviewer"] = "test fixture only"
    p = tmp_path / "review.csv"
    reviews.to_csv(p, index=False)
    result = evaluate(df, p)
    assert result["agreement"] == 1
    assert not result["research_evidence_eligible"]
    reviews.loc[0, "video_id"] = "unknown"
    reviews.to_csv(p, index=False)
    with pytest.raises(ValueError, match="not in this run"):
        evaluate(df, p)

def test_failed_run_trace(fixture_data, tmp_path):
    frame = pd.read_csv(fixture_data / "demo_videos.csv")
    frame.loc[0, "views"] = 0
    frame.to_csv(tmp_path / "invalid.csv", index=False)
    out = tmp_path / "failure"
    with pytest.raises(ValueError):
        run(tmp_path / "invalid.csv", out, fixture_data / "seed_training.csv")
    assert json.loads((out / "execution_trace.jsonl").read_text().splitlines()[-1])["stage"] == "failed"

def test_api_health_and_validation(fixture_data):
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/analyze", json={"videos": []}).status_code == 422
    rows = pd.read_csv(fixture_data / "demo_videos.csv", keep_default_na=False).to_dict("records")
    response = client.post("/analyze", json={"videos": rows})
    assert response.status_code == 200
    assert response.json()["video_count"] == 60
    rows[0]["views"] = 0
    assert client.post("/analyze", json={"videos": rows}).status_code == 422

@pytest.mark.parametrize("value,seconds", [("PT1H2M3S",3723), ("PT40S",40), ("PT2M",120)])
def test_youtube_duration(value, seconds):
    assert duration_seconds(value) == seconds

def test_youtube_invalid_duration():
    with pytest.raises(ValueError):
        duration_seconds("P1D")

def test_torch_when_installed(fixture_data):
    pytest.importorskip("torch")
    df = load(fixture_data / "demo_videos.csv")
    result = compare_models(df, True)
    assert result["pytorch"]["status"] == "executed"
    assert np.isfinite(result["pytorch"]["mae"])
