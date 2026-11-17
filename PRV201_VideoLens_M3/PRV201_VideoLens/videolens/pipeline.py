import base64
import hashlib
import html
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .data import load, CATEGORIES
from .models import classify, relations, compare_models, vision, FEATURES
from .providers import summarize


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def evaluate(df, review_path):
    if not review_path:
        return {"status": "pending", "reviewed": 0, "agreement": None,
                "note": "No human labels supplied; 80% target is not yet assessed"}
    review = pd.read_csv(review_path, dtype=str).fillna("")
    if not {"video_id", "human_category", "reviewer"}.issubset(review.columns):
        raise ValueError("Review file requires video_id, human_category, reviewer")
    review = review[review.human_category.str.strip() != ""].copy()
    if review.video_id.duplicated().any():
        raise ValueError("Duplicate reviewed IDs")
    if not set(review.video_id).issubset(set(df.video_id)):
        raise ValueError("Review contains IDs not in this run")
    if not set(review.human_category).issubset(CATEGORIES):
        raise ValueError("Unknown human_category")
    if review.reviewer.str.strip().eq("").any():
        raise ValueError("Every human label needs a reviewer")
    merged = review.merge(df[["video_id", "category", "data_origin"]], on="video_id", validate="one_to_one")
    n = len(merged)
    agreement = float((merged.human_category == merged.category).mean()) if n else None
    observed = n > 0 and merged.data_origin.eq("observed").all()
    return {"status": "executed" if n else "pending", "reviewed": n, "agreement": agreement,
            "target_met": bool(n >= 10 and agreement >= .8) if n else False,
            "research_evidence_eligible": bool(observed and n >= 10),
            "note": "Team agreement, not ground-truth accuracy. Synthetic reviews cannot establish real-video performance.",
            "comparisons": merged.to_dict(orient="records")}


def plot_charts(df, out):
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#34435c", "text.color": "#17263e", "font.size": 10})
    paths = []
    for field, label in [("duration_seconds", "Duration (seconds)"), ("hashtag_count", "Number of hashtags")]:
        fig, ax = plt.subplots(figsize=(7.2, 4))
        for group, color in [("higher", "#147d80"), ("lower_or_equal", "#72829b")]:
            part = df[df.performance_group == group]
            ax.scatter(part[field], part.engagement_rate * 100, color=color, alpha=.8, label=group.replace("_", " "))
        ax.set(xlabel=label, ylabel="Engagement per view (%)", title=f"{label} and observed engagement")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out / f"{field}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    groups = df.groupby("category").engagement_rate.agg(["median", "count"]).sort_values("median")
    ax.barh(groups.index, groups["median"] * 100, color="#147d80")
    ax.set(xlabel="Median engagement per view (%)", title="Content categories · text-model classifications")
    for i, (_, row) in enumerate(groups.iterrows()):
        ax.text(row["median"] * 100, i, f"  n={int(row['count'])}", va="center", fontsize=9)
    ax.margins(x=.18)
    fig.tight_layout()
    p = out / "categories.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def render_report(df, stats, charts, out):
    esc = html.escape
    synthetic = stats["contains_synthetic_data"]
    banner = "SYNTHETIC DEMONSTRATION — fabricated data; no real-world findings" if synthetic else "OBSERVED DATA — exploratory analysis; correlation does not establish causation"
    blocks = []
    for chart in charts:
        encoded = base64.b64encode(chart.read_bytes()).decode()
        blocks.append(f'<figure><img alt="{esc(chart.stem)} chart" src="data:image/png;base64,{encoded}"></figure>')
    relation_table = pd.DataFrame(stats["feature_relationships"]).to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
    comparison = df.groupby("performance_group").agg(videos=("video_id", "count"), median_engagement=("engagement_rate", "median"), median_duration=("duration_seconds", "median"), median_hashtags=("hashtag_count", "median")).reset_index()
    comparison.to_csv(out / "performance_comparison.csv", index=False)
    narratives = "".join(f'<h3>{esc(p["provider"])} · {esc(p["status"])}</h3><p>{esc(p.get("text", p.get("note", "")))}</p>' for p in stats["narratives"])
    models = esc(json.dumps(stats["model_comparison"], indent=2))
    review = stats["human_review"]
    review_text = f'{review["reviewed"]} reviewed; agreement: ' + (f'{review["agreement"]:.0%}' if review["agreement"] is not None else 'pending')
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VideoLens · PRV-201 Analysis Report</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f4f8;color:#17263e;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1120px;margin:40px auto;padding:0 24px}}header{{border-top:5px solid #147d80;padding-top:20px}}h1{{font-size:46px;line-height:1.1;margin:12px 0}}h2{{margin-top:36px;font-size:24px}}.eyebrow{{letter-spacing:.16em;font-size:12px;font-weight:700;color:#147d80}}.banner{{padding:12px 18px;background:#fff0cf;border:1px solid #e2bf69;border-radius:8px;font-weight:600}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card,figure,section{{background:white;border:1px solid #dce3ec;border-radius:12px;padding:20px}}.number{{font-size:32px;font-weight:700}}.muted{{color:#5b6b80}}.charts{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}figure{{margin:0}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{text-align:left;border-bottom:1px solid #e0e6ed;padding:10px}}pre{{white-space:pre-wrap;font-size:12px;overflow-wrap:anywhere}}p{{max-width:85ch}}@media(max-width:760px){{.cards,.charts{{grid-template-columns:1fr}}main{{padding:0 12px}}h1{{font-size:34px}}section{{overflow:auto}}}}@media print{{body{{background:white}}main{{margin:0}}figure,section{{break-inside:avoid}}}}
</style><main><header><div class="eyebrow">PRV-201 / TRACK A / M3 PROTOTYPE</div><h1>VideoLens</h1><p class="muted">A reproducible view of content, engagement, and the limits of the evidence.</p></header>
<div class="banner">{banner}</div><div class="cards"><div class="card"><div class="muted">Videos analyzed</div><div class="number">{len(df)}</div>YouTube · one platform</div><div class="card"><div class="muted">Median engagement</div><div class="number">{stats['median_engagement_rate']:.2%}</div>(likes + comments) / views</div><div class="card"><div class="muted">Human review</div><div class="number">{review['reviewed']} / 10</div>{esc(review_text)}</div></div>
<h2>What this run shows</h2><p>{esc(stats['computed_summary'])}</p><p>Higher performance means engagement strictly above this sample's median; ties belong to lower or equal. These groups describe this dataset and do not define future popularity.</p>
<div class="charts">{''.join(blocks)}</div><h2>Higher / lower comparison</h2><section>{comparison.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}</section>
<h2>Feature relationships</h2><section>{relation_table}<p class="muted">Spearman rank correlations. P-values are exploratory and unadjusted for multiple comparisons. Relationships may reflect creator size, video age, sampling, or other confounders.</p></section>
<h2>Model comparison</h2><section><pre>{models}</pre></section><h2>Provider narratives</h2><section>{narratives or '<p>No cloud providers requested. No external API calls made.</p>'}</section>
<h2>Limitations and next steps</h2><section><p>The text classifier learns from a small handwritten seed set. Its confidence scores are not calibrated. Review at least ten independently labeled, real videos before reporting classification agreement. No causal claims or reliable virality predictions are supported.</p><p>Frame analysis, when requested, inspects at most three supplied images per video. It does not analyze audio, pacing, or a full video. Missing likes or comments must be resolved rather than silently treated as zero. Shares are excluded because the collector does not obtain them.</p><p>Replace the synthetic fixture with at least 50 observed videos, record a sampling rule and collection dates, and document missing or excluded items. Keep your human reviewers blind to predicted categories until their labels are recorded.</p></section>
<footer><p class="muted">Generated {esc(stats['created_at'])}. See manifest.json, execution_trace.jsonl, statistics.json, and enriched_videos.csv for inspectable evidence.</p></footer></main></html>"""
    (out / "report.html").write_text(document, encoding="utf-8")


def run(input_path, output_dir, training_path, use_torch=False, frame_root=None, providers=(), review_path=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Output directory must be empty; use a new directory to preserve previous evidence")
    trace = []
    def event(stage, **details):
        trace.append({"time": datetime.now(timezone.utc).isoformat(), "stage": stage, **details})
        (out / "execution_trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in trace), encoding="utf-8")
    event("started")
    try:
        df = load(input_path)
        event("validated", rows=len(df))
        df = classify(df, training_path)
        event("text_classification", model="TF-IDF + LogisticRegression", rows=len(df))
        vision_status = {"status": "not requested", "videos_with_frames": 0}
        if frame_root:
            visual = vision(df, frame_root)
            visual.to_csv(out / "frame_classifications.csv", index=False)
            df = df.merge(visual, on="video_id", how="left", validate="one_to_one")
            vision_status = {"status": "executed", "videos_with_frames": len(visual)}
            event("vision", **vision_status)
        models = compare_models(df, use_torch)
        event("models", pytorch=models["pytorch"]["status"])
        review = evaluate(df, review_path)
        synthetic = bool(df.data_origin.eq("synthetic").any())
        rel = relations(df)
        usable = [x for x in rel if x["spearman_rho"] is not None]
        strongest = max(usable, key=lambda x: abs(x["spearman_rho"])) if usable else None
        summary = (f"Analyzed {len(df)} {'partly or wholly synthetic' if synthetic else 'observed'} videos. "
                   f"Median engagement was {df.engagement_rate.median():.2%}. " +
                   (f"The strongest of the three numeric associations was {strongest['feature']} (Spearman rho={strongest['spearman_rho']:.3f}). " if strongest else "Numeric associations could not be estimated because inputs were constant. ") +
                   "This is exploratory association, not evidence of cause or reliable prediction.")
        stats = {"created_at": datetime.now(timezone.utc).isoformat(), "video_count": len(df),
                 "contains_synthetic_data": synthetic, "median_engagement_rate": float(df.engagement_rate.median()),
                 "feature_relationships": rel, "model_comparison": models, "human_review": review,
                 "vision": vision_status, "computed_summary": summary, "narratives": []}
        for provider in providers:
            try:
                narrative = summarize(provider, {"video_count": len(df), "contains_synthetic_data": synthetic,
                    "median_engagement_rate": stats["median_engagement_rate"], "feature_relationships": rel,
                    "computed_summary": summary})
            except Exception as exc:
                # Do not log provider exception bodies: they may contain requests or credentials.
                narrative = {"provider": provider, "status": "failed", "note": f"{type(exc).__name__}: check provider configuration, access, and network"}
            stats["narratives"].append(narrative)
            event("provider", provider=provider, status=narrative["status"])
        df.drop(columns=["text"]).to_csv(out / "enriched_videos.csv", index=False)
        if not review_path:
            selected = df[["video_id", "title"]].sample(n=min(10,len(df)), random_state=201).copy()
            selected["human_category"] = ""
            selected["reviewer"] = ""
            selected.to_csv(out / "human_review_template.csv", index=False)
        save_json(out / "statistics.json", stats)
        charts = plot_charts(df, out)
        render_report(df, stats, charts, out)
        event("completed", report="report.html", contains_synthetic_data=synthetic)
        versions = {}
        for name in ["pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "torch", "transformers", "openai", "anthropic", "boto3", "fastapi"]:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        inputs = {str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in [input_path, training_path] + ([review_path] if review_path else [])}
        save_json(out / "manifest.json", {"status": "completed", "python": platform.python_version(), "versions": versions,
                  "input_sha256": inputs, "config": {"torch": use_torch, "vision": bool(frame_root), "providers": list(providers)},
                  "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},
                  "submission_ready": bool(not synthetic and len(df) >= 50 and review.get("research_evidence_eligible") and review.get("target_met")),
                  "note": "submission_ready checks data/review targets only; instructor assesses the complete rubric"})
        return stats
    except Exception as exc:
        event("failed", error_type=type(exc).__name__)
        raise
