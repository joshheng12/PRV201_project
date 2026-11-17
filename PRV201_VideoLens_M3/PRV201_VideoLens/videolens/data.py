import re
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORIES = ["education", "entertainment", "food", "fitness", "technology"]
REQUIRED = ["video_id", "platform", "title", "description", "views", "likes", "comments", "duration_seconds", "hashtags", "data_origin"]

def validate(frame):
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df = frame.copy().fillna("")
    if len(df) < 10 or len(df) > 5000:
        raise ValueError("Provide 10 to 5000 videos; M3 target is at least 50")
    for col in ["video_id", "platform", "title", "data_origin"]:
        df[col] = df[col].astype(str).str.strip()
        if (df[col] == "").any():
            raise ValueError(f"{col} cannot be empty")
    if df.video_id.duplicated().any():
        raise ValueError("Duplicate video_id values; remove duplicates before analysis")
    if set(df.platform.str.lower()) != {"youtube"}:
        raise ValueError("This prototype analyzes YouTube only")
    if not set(df.data_origin).issubset({"synthetic", "observed"}):
        raise ValueError("data_origin must be synthetic or observed")
    for col in ["views", "likes", "comments", "duration_seconds"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{col} must contain finite, nonnegative values; missing is not zero")
        if col != "duration_seconds" and (values % 1 != 0).any():
            raise ValueError(f"{col} must contain whole-number counts")
        df[col] = values
    if (df.duration_seconds <= 0).any():
        raise ValueError("duration_seconds must be positive")
    if (df.views <= 0).any():
        raise ValueError("views must be positive to calculate engagement")
    df["text"] = df.title.astype(str) + " " + df.description.astype(str)
    df["hashtag_count"] = df.hashtags.astype(str).apply(lambda x: len(re.findall(r"#\w+", x)))
    df["title_length"] = df.title.str.len()
    df["engagement_rate"] = (df.likes + df.comments) / df.views
    # Sample-relative descriptive groups, not prediction labels.
    median = float(df.engagement_rate.median())
    df["performance_group"] = np.where(df.engagement_rate > median, "higher", "lower_or_equal")
    return df

def load(path):
    return validate(pd.read_csv(path, dtype={"video_id": str}, keep_default_na=False))

def make_demo(destination):
    """Fabricated software fixture; never presented as observed YouTube evidence."""
    rng = np.random.default_rng(201)
    vocab = {
      "education": ["algebra lesson", "history explained", "science tutorial", "study skills"],
      "entertainment": ["comedy sketch", "funny challenge", "dance performance", "movie reaction"],
      "food": ["pasta recipe", "baking bread", "cooking dinner", "street food"],
      "fitness": ["workout routine", "strength training", "yoga flow", "cardio exercise"],
      "technology": ["phone review", "coding project", "computer setup", "gadget test"]}
    rows, training = [], []
    for cat, phrases in vocab.items():
        for phrase in phrases:
            for suffix in ["guide for beginners", "tips and demonstration", "step by step overview"]:
                training.append({"text": phrase + " " + suffix, "category": cat, "source": "handwritten_seed"})
    for i in range(60):
        cat = CATEGORIES[i % 5]
        duration = int(rng.integers(25, 900))
        tags = int(rng.integers(0, 7))
        views = int(rng.integers(500, 120000))
        rate = float(np.clip(.025 + .006 * (cat == "food") + .0018 * tags - .000008 * duration + rng.normal(0, .006), .003, .12))
        likes = int(views * rate * .88)
        comments = int(views * rate * .12)
        rows.append(dict(video_id=f"SYNTHETIC_{i+1:03}", platform="youtube",
          title=vocab[cat][int(rng.integers(0, 4))] + f" | demo {i+1}",
          description=f"A fabricated {cat} example for software demonstration only.",
          views=views, likes=likes, comments=comments, duration_seconds=duration,
          hashtags=" ".join(f"#demo{t}" for t in range(tags)), data_origin="synthetic",
          collected_at="2026-11-17", creator_id=f"synthetic_creator_{i%12}", seed_category=cat))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "demo_videos.csv", index=False)
    pd.DataFrame(training).to_csv(destination / "seed_training.csv", index=False)
    pd.DataFrame(columns=REQUIRED + ["creator_id", "collected_at"]).to_csv(destination / "real_videos_template.csv", index=False)
