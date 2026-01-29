import pytest
import torch
from medical_image_ai_toolkit.datamodules.tensor_datamodule import TensorDatamodule 

def test_tensor_dataset_length():
    samples = [{"x": torch.randn(1)}, {"x": torch.randn(1)}]
    ds = TensorDatamodule(samples)
    assert len(ds) == 2


def test_tensor_dataset_deterministic_sorting():
    samples = [
        {"patient_id": "b", "x": torch.tensor(1)},
        {"patient_id": "a", "x": torch.tensor(2)},
    ]
    ds = TensorDatamodule(samples, deterministic=True)
    assert ds[0]["patient_id"] == "a"

def test_tensor_dataset_getitem():
    sample = {"x": torch.tensor([1, 2, 3])}
    ds = TensorDatamodule([sample])
    out = ds[0]
    assert torch.equal(out["x"], sample["x"])

def test_tensor_dataset_empty_rejected():
    with pytest.raises(ValueError):
        TensorDatamodule([])