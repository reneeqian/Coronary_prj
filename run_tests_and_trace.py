import subprocess
from pathlib import Path
import sys

from regulatory_tools.traceability.generator import (
    generate_trace_rows,
    write_markdown,
)
from regulatory_tools.traceability.validate_traceability import (
    validate_traceability,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "tests"
REQUIREMENTS_YAML = PROJECT_ROOT / "docs" / "requirements.yaml"
EVIDENCE_ROOT = PROJECT_ROOT / "artifacts" / "evidence_runs"
OUTPUT_MATRIX = PROJECT_ROOT / "artifacts" / "traceability_matrix.md"


def run_pytest():
    print("\n[Runner] Running pytest...\n")

    result = subprocess.run(
        ["pytest", str(TEST_DIR)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print("\n[Runner] Pytest failed. Aborting traceability generation.")
        sys.exit(1)


def generate_traceability_matrix():
    print("\n[Runner] Generating traceability matrix...\n")

    # Validate requirement coverage
    missing, untracked = validate_traceability(
        requirements_yaml=REQUIREMENTS_YAML,
        test_dir=TEST_DIR,
    )

    if missing:
        print("[Traceability] Requirements declared but not tested:")
        for r in sorted(missing):
            print(f"  - {r}")

    if untracked:
        print("[Traceability] Tests reference undeclared requirements:")
        for r in sorted(untracked):
            print(f"  - {r}")

    # Generate rows from evidence
    rows = generate_trace_rows(EVIDENCE_ROOT)

    if not rows:
        print("[Traceability] No evidence artifacts found.")
        sys.exit(1)

    OUTPUT_MATRIX.parent.mkdir(exist_ok=True)
    write_markdown(rows, OUTPUT_MATRIX)

    print(f"\n[Traceability] Matrix written to: {OUTPUT_MATRIX}\n")


if __name__ == "__main__":
    run_pytest()
    generate_traceability_matrix()
