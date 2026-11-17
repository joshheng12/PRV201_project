# VideoLens — PRV-201 M3 / Track A

A working single-platform video-engagement analysis pipeline. It classifies YouTube titles/descriptions, analyzes three numeric features, compares scikit-learn and PyTorch models, optionally examines supplied frames with Hugging Face Transformers, and can generate narratives through Claude, OpenAI, AWS Bedrock, and Azure OpenAI.

**The included 60-video dataset is entirely synthetic. It demonstrates execution, not research results. Real-video collection, independent human review, and credentialed cloud runs remain your team's work before submission.**

## Start here (Windows PowerShell)

Run these commands from this folder after extracting the ZIP:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[full,dev]"
.venv/Scripts/python -m videolens.cli analyze --input data/demo_videos.csv --out results/my_demo --torch
Start-Process results/my_demo/report.html
```

Use a new empty output directory for each run. This preserves prior evidence. If you only need the lightweight offline pipeline, install `-e ".[dev]"` and omit `--torch`.

You can open `results/demo/report.html` immediately without installing anything. It is a standalone HTML report with embedded charts.

## What each stack actually does

| Stack from original proposal | Implemented role | How to execute |
|---|---|---|
| Python | Collection, orchestration, validation, statistics, reporting | Every CLI/API run |
| scikit-learn | TF-IDF + logistic text classification, training-only standardization, Ridge comparison | Every analysis |
| PyTorch | 3-input neural regression model, same held-out rows as Ridge | `--torch` |
| Hugging Face Transformers | CLIP zero-shot classification of up to three local frames per video, using PyTorch | `--frames data/frames` |
| Claude / Anthropic | Narrative explaining computed statistics | `--providers claude` |
| OpenAI API | Alternate computed-statistics narrative via Responses API | `--providers openai` |
| AWS Bedrock | Converse API narrative backend | `--providers bedrock` |
| Azure AI / Azure OpenAI | Azure OpenAI v1 narrative backend | `--providers azure` |
| API delivery | FastAPI metadata-analysis endpoint and interactive API documentation | `uvicorn videolens.api:app` |

The cloud providers are interchangeable integrations. They are not all required for every analysis; select all four to demonstrate each. They do not train or run the local models. A pipeline is the honest description: this is deterministic orchestration, not an autonomous browsing agent.

## Real-data workflow

1. Define a reproducible YouTube sampling rule before looking at engagement. Record topic, publication window, inclusion rules, collection date, and how IDs were selected. Avoid selecting only known viral examples. Video age, channel size, and topic can confound results.
2. Copy `.env.example` to `.env` and fill in your own YouTube API key. Place one YouTube video ID (not URL) per line in `data/video_ids.txt`. Collect more than 50 if some may be excluded:

```powershell
.venv/Scripts/python -m videolens.cli collect-youtube --ids data/video_ids.txt --out data/collection_01
.venv/Scripts/python -m videolens.cli analyze --input data/collection_01/videos.csv --out results/real_first --torch
```

3. Have team members label the ten videos selected in `results/real_first/human_review_template.csv`. Fill `human_category` and `reviewer`, using the category definitions in `docs/DESIGN.md`. Keep reviewers blind to model predictions. Do not copy synthetic labels or generated predictions into the human-label column.
4. Save the completed review outside the run directory and rerun:

```powershell
.venv/Scripts/python -m videolens.cli analyze --input data/collection_01/videos.csv --out results/real_reviewed --torch --reviews data/human_reviews.csv
```

At least ten reviews and 80% agreement meet the planned numerical target. Report actual disagreement even if the target is missed. Do not edit the ten labels to reach 80%.

The collector saves public metadata, raw API responses, and exclusions. It does not download media or collect shares. Unavailable metrics are excluded and recorded instead of invented. You can also populate `data/real_videos_template.csv` manually from documented observations; `data_origin` must be `observed`.

## Vision workflow

Use frames you are permitted to analyze. Create `data/frames/VIDEO_ID/` and put one to three `.jpg` or `.png` images there, preferably sampled consistently (for example, 10%, 50%, and 90% of duration). The code takes the first three filenames alphabetically. It does not extract frames or watch full videos.

```powershell
.venv/Scripts/python -m videolens.cli analyze --input data/collection_01/videos.csv --out results/real_vision --torch --frames data/frames --reviews data/human_reviews.csv
```

The first vision run downloads `openai/clip-vit-base-patch32` from Hugging Face; this is a local model, not an OpenAI API call. `HF_MODEL` can override the checkpoint. Outputs are stored separately in `frame_classifications.csv` and do not overwrite text labels. Score is a relative similarity, not a calibrated probability. No frames means no evidence of real vision execution.

## Cloud integration workflow

Fill the provider-specific model identifiers and credentials in `.env`. No identifiers are hardcoded; your selected models/deployments must support the documented API. For Azure, use a v1 base URL ending in `/openai/v1/` and a deployment that supports Responses. AWS uses boto3's normal credential chain; set `AWS_PROFILE` only when needed. Keep credentials out of the submission.

```powershell
.venv/Scripts/python -m videolens.cli analyze --input data/collection_01/videos.csv --out results/full_stack --torch --frames data/frames --reviews data/human_reviews.csv --providers claude openai bedrock azure
```

This makes up to one generation request per selected provider, plus SDK retries. Only aggregate statistics are sent to cloud narrators. Check `statistics.json` for each provider's actual `executed` or `failed` status. Provider failures are recorded while preserving local results; they do not count as successful integration evidence. Cloud outputs are commentary; numerical results come from Python. Running cloud integrations may incur your account's normal charges.

## API

```powershell
.venv/Scripts/python -m uvicorn videolens.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs. `POST /analyze` accepts `{"videos": [...]}` with 10–5000 metadata rows matching the CSV schema and returns the computed statistics. `/health` reports API status. The local API makes no cloud calls; full-stack runs use the CLI. Keep this prototype bound to localhost; it has no production authentication or rate limiter.

## Outputs

- `report.html`: readable standalone report with embedded charts; browser Print can produce a PDF.
- `enriched_videos.csv`: source metadata, classifications, features, engagement, descriptive groups.
- `statistics.json`: relationships, held-out model metrics, human review, vision/cloud status.
- `performance_comparison.csv`: higher versus lower/equal summaries.
- Three PNG charts: duration, hashtags, and category relationships.
- `human_review_template.csv`: ten randomly selected IDs, without predicted labels (when no review input supplied).
- `execution_trace.jsonl`: stage-level execution record, including failures.
- `manifest.json`: dependency versions, input/output hashes, configuration, and data/review readiness flag.

Counts of views/likes/comments are used only to define the response variable, never as predictive inputs. Preprocessing is fit only on training rows. If complete creator IDs are available and there are at least four creators, models use held-out creators; otherwise the report discloses a random holdout and its limitation.

## Verify and submit

```powershell
.venv/Scripts/python -m pytest -q
```

Read `docs/DESIGN.md`, `docs/SUBMISSION_CHECKLIST.md`, and `docs/DEMO_SCRIPT.md`. This package is repository-ready source, not a published GitHub repository. Create or use your team's repository, include the code/design/evidence, and submit its link according to course rules. Review and understand all code and disclose AI assistance if your course requires it.
