# Coronary Artery Calcium (CAC) Detection – Engineering Demonstration Project

[![Coronary CI](https://github.com/reneeqian/Coronary_prj/actions/workflows/run-tests.yml/badge.svg)](https://github.com/reneeqian/Coronary_prj/actions/workflows/run-tests.yml)

Application example built on `medical_image_ai_toolkit` for coronary calcium detection from gated CT. Engineering demonstration — not a clinical product.

## Install

```bash
pip install -e ../regulatory_tools
pip install -e ../medical_image_ai_toolkit
pip install -e .
```

## Dataset

Tests that require real data use the [COCA dataset](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa). Place the extracted dataset at:

```
data/raw/coca/cocacoronarycalciumandchestcts-2/Gated_release_final/
```

Tests that need real data are skipped automatically if this path doesn't exist.

## Tests

```bash
python -m pytest          # unit + synthetic tests (no dataset required)
python runtests.py        # full suite + traceability matrix + forge health report
```

Smoke tests require a completed training run (`artifacts/training_runs/model.pt` + `partitions.json`):

```bash
python scripts/smoketesttraining.py
python scripts/smoketestmodeltesting.py
```

## Status Report

Print a one-page console summary of the latest training, tuning, and model testing runs:

```bash
python scripts/status_report.py
```

Or from Python:

```python
from Coronary_prj.reporting import status_report
status_report()
```

Looks for run artifacts under `artifacts/training_runs/`, `artifacts/tuning_runs/`,
and `artifacts/model_testing_runs/`. Prints "no runs found" for any missing category.

To export a PDF from a specific run:

```python
from medical_image_ai_toolkit.reporting import generate_training_pdf
pdf = generate_training_pdf("artifacts/training_runs")
print("Report written to:", pdf)
```

## Layout

- `src/coronary_prj/ingestors/` — COCA dataset ingestion (`COCAGatedIngestor`)
- `src/coronary_prj/task_definitions/` — `CoronaryCalciumTask` (slice extraction, loss)
- `src/coronary_prj/models/` — `UNet2D`, `SmallSegmentationCNN`
- `docs/requirements.yaml` — project requirements (`SYS-`, `DAT-`, `TRN-` prefixes)

---

## Forge Health

Latest report: see the [Actions tab](../../actions) or the job summary on any PR's Checks tab.

<!-- forge-health-start -->
*Last run: 2026-06-09*

**Grade: A** (score: 0.90)

| Collector | Score |
|-----------|-------|
| Test Metrics | 0.93 |
| Complexity | 0.70 |
| Dependency Health | 0.85 |
| Requirements Coverage | 1.00 |
| Static Analysis | 0.99 |
| Type Coverage | 1.00 |
| Dead Code | 1.00 |
<!-- forge-health-end -->
