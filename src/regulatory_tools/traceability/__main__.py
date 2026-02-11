from pathlib import Path
from src.regulatory_tools.traceability.generator import generate_trace_rows, write_markdown

def main():
    rows = generate_trace_rows(
        evidence_root=Path("artifacts/evidence_runs")
    )
    write_markdown(
        rows,
        output=Path("docs/traceability.md")
    )

if __name__ == "__main__":
    main()
