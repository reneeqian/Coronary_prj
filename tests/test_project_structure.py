import pytest
from pathlib import Path

from regulatory_tools.evidence.evidence_report import EvidenceReport


@pytest.mark.requirement("DOC-001")
@pytest.mark.requirement("DOC-002")
def test_required_project_documentation_exists(
    request,
    evidence_output_dir,
):
    """
    Verifies that:
      - docs/requirements.yaml exists
      - README.md exists at project root
    """

    project_root = Path(__file__).resolve().parents[1]
    requirements_path = project_root / "docs" / "requirements.yaml"
    readme_path = project_root / "README.md"

    report = EvidenceReport(
        subject="Project Documentation → Required Files Presence",
        test_id=request.node.nodeid,
    )

    # ==============================
    # requirements.yaml Check
    # ==============================

    report.info(
        message="Checking for machine-readable requirements.yaml",
        requirement_id="DOC-001",
        context=str(requirements_path),
    )

    if not requirements_path.exists():
        report.error(
            message="requirements.yaml not found in docs directory",
            requirement_id="DOC-001",
        )

    # ==============================
    # README.md Check
    # ==============================

    report.info(
        message="Checking for README.md in project root",
        requirement_id="DOC-002",
        context=str(readme_path),
    )

    if not readme_path.exists():
        report.error(
            message="README.md not found in project root",
            requirement_id="DOC-002",
        )

    # ==============================
    # Save Evidence
    # ==============================

    report.auto_save("project_documentation_presence", evidence_output_dir)

    assert not report.has_errors, report.summary()