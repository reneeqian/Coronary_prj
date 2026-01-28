from pathlib import Path
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestors.coca_gated_ingestor import COCAGatedIngestor
from src.medimg_training.src.adapters.patient_sample_to_tensor import PatientSampleTensorAdapter


def test_tensor_adapter_output_shapes():
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
