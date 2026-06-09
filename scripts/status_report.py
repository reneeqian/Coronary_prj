"""
CLI entry point for the Coronary_prj status report.

Usage::

    python scripts/status_report.py

Prints a console summary of the latest training, hyperparameter tuning,
and model testing run metrics found under ``artifacts/``.
"""

from __future__ import annotations

from pathlib import Path

from Coronary_prj.reporting import status_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    status_report(project_root=PROJECT_ROOT)
