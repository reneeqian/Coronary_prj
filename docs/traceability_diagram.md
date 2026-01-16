```mermaid
graph LR
  FR001[FR-001 Ingest Data] --> D1[coronary_ct_dataset.py]
  D1 --> T1[test_dataset.py]

  FR002[FR-002 Validate Data] --> V1[validators.py]
  V1 --> T2[test_validators.py]
  ```
