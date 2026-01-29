import subprocess
from datetime import datetime
from pathlib import Path

EVIDENCE_DIR = Path("artifacts/evidence_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

print(f"[Runner] Evidence output → {EVIDENCE_DIR}")

env = {
    **dict(**__import__("os").environ),
    "EVIDENCE_OUTPUT_DIR": str(EVIDENCE_DIR),
}

result = subprocess.run(
    ["pytest", "-s", "tests", "src/medical_image_ai_toolkit/tests"],
    env=env,
)

if result.returncode != 0:
    raise SystemExit("Tests failed")

print("[Runner] Test run complete")
