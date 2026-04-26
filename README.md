# Coronary Artery Calcium (CAC) Detection – Engineering Demonstration Project

[![Coronary CI](https://github.com/reneeqian/Coronary_prj/actions/workflows/run-tests.yml/badge.svg)](https://github.com/reneeqian/Coronary_prj/actions/workflows/run-tests.yml)

Application example built on top of `medical_image_ai_toolkit` for coronary
calcium detection from gated CT data.

This project is an engineering demonstration. It is not a clinical product,
and the goal is disciplined structure rather than model performance.

## Mission

This repository exists to show how a concrete medical imaging project can keep:

- dataset-specific ingestion logic in project code
- reusable training and validation logic in the toolkit
- machine-readable requirements and executable tests close together
- documentation lightweight and stable

## Scope

The project focuses on:

- deterministic ingestion of the coronary dataset
- generation of training targets for the coronary calcium task
- verification evidence and traceability around that project-specific code

It does not aim to document a full clinical workflow or deployment system.

## Repository Layout

- `src/Coronary_prj/`: coronary-specific ingestors and task definitions
- `docs/requirements.yaml`: stable project requirements
- `tests/`: executable verification and evidence capture
- `runtests.py`: project verification entry point

## Verification

Run:

```bash
python runtests.py
```

This runs the project test suite, coverage, and traceability generation.

## Documentation Approach

The primary documentation for this project is:

- this README for mission and boundaries
- `docs/requirements.yaml` for behavioral expectations
- tests for executable examples of project behavior

Requirements follow the convention defined in
`regulatory_tools/docs/Requirements_Convention.md`.

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

