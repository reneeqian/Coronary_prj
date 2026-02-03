import subprocess
from datetime import datetime
from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

assert Path("src").exists(), "Must run from repo root"
assert Path("tests").exists(), "tests/ directory missing"

EVIDENCE_DIR = Path("artifacts/evidence_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

print(f"[Runner] Repo root → {REPO_ROOT}")
print(f"[Runner] Evidence output → {EVIDENCE_DIR}")

env = {
    **os.environ,
    "EVIDENCE_OUTPUT_DIR": str(EVIDENCE_DIR),
}

result = subprocess.run(
    [
        "pytest",
        "-s",
        "tests",
        "src/medical_image_ai_toolkit/tests",
    ],
    env=env,
)

if result.returncode != 0:
    sys.exit("Tests failed")

print("[Runner] Test run complete")
