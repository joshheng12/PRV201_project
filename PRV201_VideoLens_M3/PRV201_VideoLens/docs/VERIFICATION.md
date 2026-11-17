# Verification record â€” November 17, 2026

## Executed successfully

- Synthetic demo: 60 rows, one platform, engagement/features, text classifications, three charts, group table, HTML report, trace and manifest.
- scikit-learn TF-IDF/logistic classification and Ridge regression.
- PyTorch neural regression on the same creator-held-out evaluation rows (45 train / 15 test).
- Full pytest suite: **29 passed in 28.80 seconds**. Five dependency/pandas deprecation or future warnings; no test failures.
- FastAPI health, schema validation, and complete analysis response through in-process TestClient.
- Duplicate/invalid metrics, absent columns, constant feature handling, held-out creator separation, failure traces, review integrity, output preservation, synthetic-evidence disclosure, and YouTube duration parsing.
- Mocked contracts for Claude, OpenAI, Azure OpenAI, and Bedrock; mocked frame aggregation and frame-path rejection.

The Windows sandbox blocked asyncio's local socket initialization; the test suite was rerun outside that restriction and completed. Tests do not expose a public server or make live cloud requests.

- Actual installed Transformers pipeline, Anthropic SDK, and boto3 imports succeeded. OpenAI SDK 2.54.0 exposes Responses successfully. No provider requests were made.

## Not claimed as executed

- Live YouTube collection: requires the team's API key and selected video IDs.
- Classification agreement on ten human-reviewed observed videos: no human labels supplied.
- Live Hugging Face CLIP inference: no actual project frames supplied, and model weights not downloaded.
- Live Claude, OpenAI, AWS Bedrock, or Azure OpenAI generations: no account credentials supplied.

Integration code and mock tests establish implementation paths, not successful service authentication, model availability, or real-world quality. See the submission checklist before claiming all planned stacks have run.

## Evidence location

`results/demo/` contains the actual completed synthetic run. Its manifest captures the versions at execution time. Dependencies subsequently updated for integration compatibility may differ from that run's versions; this does not change its recorded local-model results.

The synthetic dataset intentionally contains constructed patterns, so its statistics are not research findings. Human review remains pending rather than a fabricated 80% agreement result.
