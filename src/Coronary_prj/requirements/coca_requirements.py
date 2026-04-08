from regulatory_tools.requirements.requirement_provider import RequirementProvider
from regulatory_tools.requirements.requirement_keys import RequirementKeys


class CoronaryRequirements(RequirementProvider):

    def __init__(self):
        self._mapping = {
            RequirementKeys.DATASET_VALIDATION: "DAT-001",
            RequirementKeys.PATIENT_LOAD_FAILURE: "DAT-005",
            RequirementKeys.PATIENT_SAMPLE_INVALID: "DAT-005",
            RequirementKeys.DATASET_EMPTY: "DAT-001",
        }

    def get(self, key: str) -> str | None:
        return self._mapping.get(key)