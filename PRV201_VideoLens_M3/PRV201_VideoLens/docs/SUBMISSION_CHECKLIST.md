# M3 submission checklist — evidence over claims

## Rubric mapping

| Criterion | Points | Artifact / demonstration |
|---|---:|---|
| M3.1 core behavior | 20 | Completed analysis, enriched dataset, reports, classifications, statistics |
| M3.2 scope completeness | 15 | DESIGN.md alignment table and explicit deferred items |
| M3.3 execution evidence | 25 | Runnable repository, execution_trace.jsonl, manifest.json, actual demo/results |
| M3.4 engineering rigor | 20 | Validation, tests, errors, held-out preprocessing, limitations |
| M3.5 alignment | 10 | Architecture, stack roles, change explanations |
| M3.6 documentation | 10 | README setup/run instructions and DESIGN.md |

## Before your team submits

- [ ] Read, understand, and customize the implementation; follow your course's AI-assistance rules.
- [ ] Place the project in your team's code repository and verify instructions from a clean environment.
- [ ] Collect at least 50 observed videos from YouTube; save sampling protocol, dates, exclusions, raw responses.
- [ ] Label ten actual videos independently; report agreement honestly, including failures.
- [ ] Run PyTorch and preserve its actual results.
- [ ] Supply actual permitted frames, execute Transformers, preserve frame classifications.
- [ ] Configure Claude, OpenAI, AWS Bedrock, and Azure OpenAI; execute all four and verify each says `executed`.
- [ ] Preserve a full run's trace, manifest, charts, enriched CSV, statistics, and report.
- [ ] Update DESIGN.md to describe what your team actually executed. Integration code is not live execution evidence.
- [ ] Record a short demo following DEMO_SCRIPT.md.
- [ ] Submit repository link and design document, plus demo/report as allowed by your instructor.
- [ ] Remove all `.env` files, tokens, private data, virtual environments, caches, and model downloads from your submission.

## Current package status

The bundled report is explicitly synthetic. No human-review result, real-video sample, cloud account access, or full-stack live success is fabricated. See VERIFICATION.md for exactly what was tested in this environment. This is a first-version implementation and design draft; completing the real-data and live-integration evidence is necessary to claim your full planned scope.
