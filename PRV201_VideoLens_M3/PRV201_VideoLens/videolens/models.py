import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from .data import CATEGORIES

FEATURES = ["duration_seconds", "hashtag_count", "title_length"]

def classify(df, training_path):
    seed = pd.read_csv(training_path).dropna(subset=["text", "category"])
    if set(seed.category) != set(CATEGORIES):
        raise ValueError("Training data must cover exactly the five documented categories")
    if seed.groupby("category").size().min() < 2:
        raise ValueError("Provide at least two training examples per category")
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000, random_state=201))
    model.fit(seed.text, seed.category)
    probabilities = model.predict_proba(df.text)
    df = df.copy()
    df["category"] = model.predict(df.text)
    df["category_confidence"] = probabilities.max(axis=1)
    df["classification_source"] = "sklearn_text_seed_classifier"
    return df

def relations(df):
    results = []
    for name in FEATURES:
        if df[name].nunique() < 2 or df.engagement_rate.nunique() < 2:
            results.append({"feature": name, "spearman_rho": None, "p_value": None, "status": "constant input"})
        else:
            rho, p = spearmanr(df[name], df.engagement_rate)
            results.append({"feature": name, "spearman_rho": float(rho), "p_value": float(p), "status": "exploratory; unadjusted p-value"})
    return results

def compare_models(df, use_torch=False):
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df.engagement_rate.to_numpy(dtype=np.float32)
    if "creator_id" in df and df.creator_id.astype(str).str.strip().ne("").all() and df.creator_id.nunique() >= 4:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=201).split(X, y, df.creator_id))
        method = "held-out creators (GroupShuffleSplit, seed 201)"
    else:
        train, test = train_test_split(np.arange(len(df)), test_size=.25, random_state=201)
        method = "random holdout (seed 201); creator overlap may bias results"
    scaler = StandardScaler().fit(X[train])
    a, b = scaler.transform(X[train]), scaler.transform(X[test])
    def score(pred):
        return {"mae": float(mean_absolute_error(y[test], pred)), "r2": float(r2_score(y[test], pred)) if np.ptp(y[test]) else None}
    baseline = np.full(len(test), y[train].mean())
    ridge = Ridge(alpha=1.).fit(a, y[train])
    result = {"features": FEATURES, "split": method, "train_count": len(train), "test_count": len(test),
              "train_video_ids": df.iloc[train].video_id.tolist(), "test_video_ids": df.iloc[test].video_id.tolist(),
              "mean_baseline": score(baseline), "sklearn_ridge": score(ridge.predict(b)),
              "pytorch": {"status": "not requested"},
              "interpretation": "Exploratory holdout estimate, not a validated virality predictor. Engagement counts are excluded from predictors."}
    if use_torch:
        import torch
        torch.manual_seed(201)
        torch.set_num_threads(1)
        net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=.01)
        # Standardize target using training data only for stable neural optimization.
        mu, sigma = float(y[train].mean()), max(float(y[train].std()), 1e-6)
        tx, ty = torch.tensor(a), torch.tensor((y[train] - mu) / sigma).reshape(-1, 1)
        net.train()
        for _ in range(250):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(net(tx), ty)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(b)).numpy().ravel() * sigma + mu
        result["pytorch"] = {"status": "executed", "architecture": "3 -> 8 ReLU -> 1", "epochs": 250, **score(pred)}
    return result

def vision(df, frame_root):
    from transformers import pipeline
    from PIL import Image
    root = Path(frame_root).resolve()
    detector = pipeline(task="zero-shot-image-classification", model=os.getenv("HF_MODEL", "openai/clip-vit-base-patch32"), device=-1)
    rows = []
    labels = ["an educational scene", "an entertainment performance", "food and cooking", "fitness and exercise", "technology and gadgets"]
    for video_id in df.video_id:
        # Folder names must be literal IDs, never paths.
        if Path(video_id).name != video_id or video_id in (".", "..") or "/" in video_id or "\\" in video_id:
            raise ValueError("Unsafe video ID for frame lookup")
        folder = (root / video_id).resolve()
        if root not in folder.parents:
            raise ValueError("Frame folder escapes frame root")
        frames = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[:3]
        if not frames:
            continue
        scores = np.zeros(5)
        for path in frames:
            if root not in path.resolve().parents:
                raise ValueError("Frame path escapes frame root")
            with Image.open(path) as image:
                output = detector(image.convert("RGB"), candidate_labels=labels)
            for item in output:
                scores[labels.index(item["label"])] += item["score"] / len(frames)
        rows.append({"video_id": video_id, "vision_category": CATEGORIES[int(scores.argmax())],
                     "vision_score": float(scores.max()), "frame_count": len(frames)})
    return pd.DataFrame(rows, columns=["video_id", "vision_category", "vision_score", "frame_count"])
