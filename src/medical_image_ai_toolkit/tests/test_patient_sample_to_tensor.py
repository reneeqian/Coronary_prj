from pathlib import Path
import torch
import sys
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medical_image_ai_toolkit.adapters.patient_sample_to_tensor import PatientSampleTensorAdapter
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample

@pytest.fixture
def sample():
    return PatientSample(
        patient_id="TEST-001",
        image_volume=np.zeros((16, 64, 64), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        annotations=np.zeros((16, 64, 64), dtype=np.int64),
    )

def make_invalid_sample():
    return PatientSample(
        patient_id="INVALID-001",
        image_volume=None,  # invalid
        spacing=(1.0, 1.0, 1.0),
        annotations=None,
    )


def test_tensor_adapter_output_shapes(sample):
    adapter = PatientSampleTensorAdapter(require_annotations=True)
    out = adapter(sample)

    image = out["image"]
    target = out["target"]

    assert image.ndim == 4
    assert image.shape[0] == 1
    assert image.dtype == torch.float32
    assert torch.all(image >= 0)
    assert torch.all(image <= 1)
    assert target is not None

def test_adapter_fails_on_invalid_sample():
    sample = make_invalid_sample()
    adapter = PatientSampleTensorAdapter()

    with pytest.raises(ValueError):
        adapter(sample)
