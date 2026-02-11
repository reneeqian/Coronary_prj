# Risk Management File (Demonstrative)

## 1. Purpose

This document demonstrates how a risk management file might be structured
under ISO 14971 for a SaMD product.

This project is not a medical device.

---

# 2. Hazard Identification (Demonstration Only)

| Hazard | Cause | Potential Harm (Hypothetical) | Mitigation |
|--------|-------|------------------------------|------------|
| Missed CAC detection | Model underperformance | Incorrect clinical interpretation | Validation dataset testing |
| False positives | Threshold miscalibration | Unnecessary concern | Deterministic testing |
| Incorrect slice ordering | DICOM metadata error | Incorrect scoring | Explicit sorting logic |
| Missing dataset | User error | Execution failure | Explicit exception |

---

# 3. Software Risk Controls (Demonstrated)

- Deterministic ordering
- Contract validation
- Explicit exceptions
- Traceable verification tests

---

# 4. Residual Risk

Since this is not used clinically, residual risk is theoretical.

---

# 5. Educational Objective

Demonstrate structured hazard analysis and risk documentation discipline.
