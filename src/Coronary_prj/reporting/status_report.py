"""
Coronary_prj status report — console summary of the latest runs.

Reads training_report.json, tuning_report.json, and
model_testing_report.json from their respective artifact directories
and prints a formatted summary of the most recent metrics for each
run type.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def status_report(project_root: Path | str | None = None) -> None:
    """
    Print a structured console summary of the latest run metrics.

    Looks for run artifacts under::

        {project_root}/artifacts/training_runs/
        {project_root}/artifacts/tuning_runs/
        {project_root}/artifacts/model_testing_runs/

    Parameters
    ----------
    project_root:
        Root directory of the Coronary_prj package.  Defaults to the
        package directory derived from this file's location.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
    project_root = Path(project_root)

    artifacts = project_root / "artifacts"

    sep = "=" * 62
    print(sep)
    print(f"  Coronary_prj Status Report  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(sep)

    # Training and tuning live inside Coronary_prj/artifacts/
    _print_training_section(artifacts / "training_runs")
    _print_tuning_section(artifacts / "tuning_runs")

    # Model testing may be in Coronary_prj/artifacts/model_testing_runs/ (current),
    # Coronary_prj/artifacts/validation_runs/ or the parent project root's artifacts/
    # (legacy paths from earlier pipeline versions).
    parent_artifacts = project_root.parent / "artifacts"
    _print_model_testing_section(
        artifacts / "model_testing_runs",
        fallback_roots=[
            artifacts / "validation_runs",
            parent_artifacts / "model_testing_runs",
            parent_artifacts / "validation_runs",
        ],
    )

    print(sep)


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------


def _find_latest(root: Path, report_filename: str = "") -> Path | None:
    if not root.exists():
        return None
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if report_filename:
        # skip empty or incomplete run dirs that lack the report JSON
        dirs = [d for d in dirs if (d / report_filename).exists()]
    return dirs[-1] if dirs else None


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _fmt(v, decimals: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    if v is None:
        return "—"
    return str(v)


def _row(label: str, value) -> None:
    print(f"  {label:<18}{_fmt(value) if isinstance(value, float) else (value or '—')}")


def _print_training_section(runs_root: Path) -> None:
    print()
    run_dir = _find_latest(runs_root, "training_report.json")
    if run_dir is None:
        print("TRAINING  (no runs found)")
        return

    data = _load_json(run_dir / "training_report.json") or {}
    metrics = data.get("metrics", {})
    config = data.get("config", {})
    ds = data.get("datasource", {})

    epochs_trained = metrics.get("epochs_trained", metrics.get("num_epochs"))
    epochs_max = config.get("epochs")
    epochs_str = f"{epochs_trained} / {epochs_max}" if epochs_max else str(epochs_trained)

    train_ids = ds.get("train_ids", [])
    val_ids = ds.get("val_ids", [])
    test_ids = ds.get("test_ids", [])
    patients_str = (
        f"{len(train_ids)} train  /  {len(val_ids)} val  /  {len(test_ids)} test"
        if isinstance(train_ids, list)
        else "—"
    )

    print(f"TRAINING  ({run_dir.name})")
    _row("learning_rate", config.get("learning_rate"))
    _row("epochs", epochs_str)
    _row("final_loss", metrics.get("final_loss"))
    _row("patients", patients_str)


def _print_tuning_section(runs_root: Path) -> None:
    print()
    run_dir = _find_latest(runs_root, "tuning_report.json")
    if run_dir is None:
        print("HYPERPARAMETER TUNING  (no runs found)")
        return

    data = _load_json(run_dir / "tuning_report.json") or {}
    best = data.get("best_trial") or {}
    params = best.get("params", {})
    best_loss = best.get("best_val_loss") or best.get("final_loss")

    print(f"HYPERPARAMETER TUNING  ({run_dir.name})")
    _row("trials", data.get("num_trials"))
    _row("best trial", best.get("trial_id"))
    _row("best val loss", best_loss)
    for k, v in params.items():
        _row(f"  {k}", v)


def _print_model_testing_section(
    runs_root: Path,
    fallback_roots: list[Path] | None = None,
) -> None:
    print()
    run_dir = _find_latest(runs_root, "model_testing_report.json")
    report_filename = "model_testing_report.json"

    # Walk fallback locations (legacy naming conventions)
    if run_dir is None and fallback_roots:
        for fb_root in fallback_roots:
            candidate = _find_latest(fb_root, "model_testing_report.json")
            if candidate:
                run_dir = candidate
                break
            candidate = _find_latest(fb_root, "validation_report.json")
            if candidate:
                run_dir = candidate
                report_filename = "validation_report.json"
                break

    if run_dir is None:
        print("MODEL TESTING  (no runs found)")
        return

    data = _load_json(run_dir / report_filename) or {}
    metrics = data.get("metrics", {})

    print(f"MODEL TESTING  ({run_dir.name})")
    _row("dice", metrics.get("dice"))
    _row("iou", metrics.get("iou"))
    _row("precision", metrics.get("precision"))
    _row("recall", metrics.get("recall"))
    _row("mean_loss", metrics.get("mean_loss"))
    _row("test patients", metrics.get("num_test_patients"))
