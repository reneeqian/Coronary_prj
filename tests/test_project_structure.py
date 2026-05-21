from pathlib import Path

import pytest
import yaml
from regulatory_tools.evidence.evidence_report import EvidenceReport
from regulatory_tools.quality.soup_checker import check_soup_inventory


@pytest.mark.requirement("DOC-002")
def test_requirements_yaml_valid_structure(evidence_output_dir):
    """docs/requirements.yaml exists, parses, has metadata.project and required domain prefixes."""
    project_root = Path(__file__).resolve().parents[1]
    requirements_path = project_root / "docs" / "requirements.yaml"

    report = EvidenceReport(subject="DOC-002: requirements.yaml exists and has required structure")

    if not requirements_path.exists():
        report.error("requirements.yaml not found in docs directory", "DOC-002")
        report.auto_save("DOC002_requirements_yaml_valid_structure", evidence_output_dir)
        assert not report.has_errors, report.summary()
        return

    with open(requirements_path, "r") as f:
        data = yaml.safe_load(f)

    if not data.get("metadata", {}).get("project"):
        report.error("metadata.project field missing or empty", "DOC-002")

    requirements = data.get("requirements")
    if not requirements or not isinstance(requirements, list):
        report.error("requirements list missing or empty", "DOC-002")
    else:
        prefixes_present = {req.get("id", "").split("-")[0] for req in requirements if "-" in req.get("id", "")}
        required_prefixes = {"SYS", "DAT", "VER", "DOC"}
        missing = required_prefixes - prefixes_present
        if missing:
            report.error(f"Missing required requirement categories: {sorted(missing)}", "DOC-002")
        else:
            report.info(
                f"requirements.yaml parsed; {len(requirements)} requirements covering required prefixes {sorted(required_prefixes)}",
                "DOC-002",
            )

    report.auto_save("DOC002_requirements_yaml_valid_structure", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DOC-001")
def test_readme_exists_with_heading(evidence_output_dir):
    """README.md exists in the project root and contains at least one Markdown heading."""
    project_root = Path(__file__).resolve().parents[1]
    readme_path = project_root / "README.md"

    report = EvidenceReport(subject="DOC-001: README.md exists and contains at least one Markdown heading")

    if not readme_path.exists():
        report.error("README.md not found in project root", "DOC-001")
    else:
        content = readme_path.read_text(encoding="utf-8")
        has_heading = any(line.strip().startswith("#") for line in content.splitlines())
        if not has_heading:
            report.error("README.md does not contain any Markdown headings", "DOC-001")
        else:
            report.info("README.md exists and contains at least one Markdown heading", "DOC-001")

    report.auto_save("DOC001_readme_exists_with_heading", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DOC-001")
def test_soup_inventory_complete(
    request,
    evidence_output_dir,
):
    """Verify docs/soup.yaml exists and covers all declared pyproject.toml dependencies."""
    project_root = Path(__file__).resolve().parents[1]

    report = EvidenceReport(
        subject="SOUP Inventory → All Dependencies Declared",
        test_id=request.node.nodeid,
    )

    result = check_soup_inventory(project_root)

    if not result["found"]:
        report.error(
            "docs/soup.yaml not found — all third-party dependencies must be inventoried",
            "DOC-001",
            context=str(project_root / "docs" / "soup.yaml"),
        )
    else:
        report.info(
            f"soup.yaml found: {result['soup_path']}",
            "DOC-001",
        )
        if result["unlisted_deps"]:
            report.error(
                "Dependencies in pyproject.toml are missing from soup.yaml: "
                + str(result["unlisted_deps"]),
                "DOC-001",
            )
        else:
            report.info("All pyproject.toml dependencies are listed in soup.yaml", "DOC-001")

    report.auto_save("soup_inventory_complete", evidence_output_dir)
    assert not report.has_errors, report.summary()
