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

## Layout

- `src/coronary_prj/ingestors/` — COCA dataset ingestion (`COCAGatedIngestor`)
- `src/coronary_prj/task_definitions/` — `CoronaryCalciumTask` (slice extraction, loss)
- `src/coronary_prj/models/` — `UNet2D`, `SmallSegmentationCNN`
- `docs/requirements.yaml` — project requirements (`SYS-`, `DAT-`, `TRN-` prefixes)

---

## Forge Health

<!-- forge-health-start -->
*Last run: 2026-04-26*

**Grade: B** (score: 0.89)

| Collector | Score |
|-----------|-------|
| Test Metrics | 0.94 |
| Complexity | 0.76 |
| Dependency Health | 0.85 |
| Requirements Coverage | 1.00 |
| Static Analysis | 0.83 |
| Type Coverage | 0.98 |
<!-- forge-health-end -->
