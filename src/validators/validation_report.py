# src/validators/validation_report.py

from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationIssue:
    level: str            # "ERROR" | "WARN" | "INFO"
    message: str
    context: str | None = None


@dataclass
class ValidationReport:
    subject: str
    issues: List[ValidationIssue] = field(default_factory=list)

    def error(self, message: str, context: str | None = None):
        self.issues.append(ValidationIssue("ERROR", message, context))

    def warn(self, message: str, context: str | None = None):
        self.issues.append(ValidationIssue("WARN", message, context))

    def info(self, message: str, context: str | None = None):
        self.issues.append(ValidationIssue("INFO", message, context))

    @property
    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)

    def summary(self) -> str:
        lines = [f"Validation report for {self.subject}"]
        for i in self.issues:
            prefix = f"[{i.level}]"
            ctx = f" ({i.context})" if i.context else ""
            lines.append(f"{prefix} {i.message}{ctx}")
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        print("\n=== Validation Report ===")
        print(f"Subject: {self.subject}")

        errors = [i for i in self.issues if i.level == "ERROR"]
        warnings = [i for i in self.issues if i.level == "WARNING"]
        infos = [i for i in self.issues if i.level == "INFO"]

        print(f"Errors:   {len(errors)}")
        print(f"Warnings: {len(warnings)}")
        print(f"Info:     {len(infos)}")

        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  ❌ {e.message}")
                if e.context:
                    print(f"     ↳ {e.context}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠️  {w.message}")
                if w.context:
                    print(f"     ↳ {w.context}")

        if infos:
            print("\nInfo:")
            for i in infos:
                print(f"  ℹ️  {i.message}")
                if i.context:
                    print(f"     ↳ {i.context}")

        print("\n=== End Validation Report ===\n")
