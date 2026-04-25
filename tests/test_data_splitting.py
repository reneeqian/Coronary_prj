import numpy as np
import pytest
from medical_image_ai_toolkit.dataobjects.annotation_bundle import AnnotationBundle, VectorROI
from medical_image_ai_toolkit.dataobjects.datasources.deterministic_split import (
    DeterministicHoldoutSplit,
)
from medical_image_ai_toolkit.dataobjects.datasources.medical_image_datasource import (
    MedicalImageDataSource,
)
from medical_image_ai_toolkit.dataobjects.patient_sample import PatientSample
from regulatory_tools.evidence.evidence_report import EvidenceReport

from Coronary_prj.task_definitions.coronary_calcium_task import CoronaryCalciumTask


class _SyntheticIngestor:
    def __init__(self, num_patients: int = 10) -> None:
        self.patient_ids = [f"P{i}" for i in range(num_patients)]

    def list_patient_ids(self):
        return self.patient_ids

    def load_patient_sample(self, patient_id: str) -> PatientSample:
        rng = np.random.default_rng(abs(hash(patient_id)) % (2**32))
        volume = rng.uniform(-200, 400, size=(4, 32, 32)).astype(np.float32)
        contour = np.array([[8, 8], [24, 8], [24, 24], [8, 24]], dtype=np.float32)
        roi = VectorROI(slice_index=2, contour_px=contour, label="lesion")
        annotations = AnnotationBundle(
            vector_rois={2: [roi]}, segmentation_masks=None, label_map={"lesion": 1}
        )
        return PatientSample(
            image_volume=volume,
            spacing=(1.5, 0.7, 0.7),
            annotations=annotations,
            patient_id=patient_id,
            metadata={},
        )

    def get_sample(self, patient_id: str):
        sample = self.load_patient_sample(patient_id)
        volume = sample.image_volume
        img = volume[0][None, None, :, :].astype(np.float32)
        label = np.array([[1.0]], dtype=np.float32)
        return img, label


def _make_datasource(tmp_path):
    return MedicalImageDataSource(
        dataset_root=tmp_path,
        ingestor=_SyntheticIngestor(num_patients=10),
    )


@pytest.mark.requirement("SYS-002")
@pytest.mark.requirement("DAT-011")
def test_deterministic_holdout_split_generates_three_partitions(evidence_output_dir):
    report = EvidenceReport(subject="Deterministic holdout split — three-partition generation")

    split = DeterministicHoldoutSplit(train=0.7, val=0.15, seed=42)
    ids = [f"P{i}" for i in range(20)]
    train_ids, val_ids, test_ids = split.split(ids)

    all_ids = set(train_ids) | set(val_ids) | set(test_ids)
    combined = list(train_ids) + list(val_ids) + list(test_ids)

    if len(all_ids) != len(ids):
        report.error("Partition sets overlap — patients appear in multiple partitions", "SYS-002")
    if len(combined) != len(ids):
        report.error("Total partition count does not equal full patient count", "SYS-002")
    if len(train_ids) == 0:
        report.error("Train partition is empty", "DAT-011")
    if len(test_ids) == 0:
        report.error("Test partition is empty", "DAT-011")

    report.auto_save("SYS002_DAT011_split_generates_partitions", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("SYS-002")
def test_split_is_reproducible_with_same_seed(evidence_output_dir):
    report = EvidenceReport(subject="Deterministic holdout split — seed reproducibility")

    ids = [f"P{i}" for i in range(30)]
    split_a = DeterministicHoldoutSplit(seed=42)
    split_b = DeterministicHoldoutSplit(seed=42)

    train_a, val_a, test_a = split_a.split(ids)
    train_b, val_b, test_b = split_b.split(ids)

    if train_a != train_b or val_a != val_b or test_a != test_b:
        report.error("Same seed produces different splits — split is not deterministic", "SYS-002")

    report.auto_save("SYS002_split_reproducibility", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("DAT-011")
def test_datasource_partition_assignment(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Datasource partition generation")

    ds = _make_datasource(tmp_path)
    split = DeterministicHoldoutSplit(train=0.7, val=0.15, seed=0)
    ds.create_partitions(split)

    if not ds.has_partitions():
        report.error("Datasource reports no partitions after create_partitions()", "DAT-011")
    else:
        if len(ds.get_train_ids()) == 0:
            report.error("Train partition is empty", "DAT-011")
        if len(ds.get_test_ids()) == 0:
            report.error("Test partition is empty", "DAT-011")

    report.auto_save("DAT011_datasource_partition_assignment", evidence_output_dir)
    assert not report.has_errors, report.summary()


@pytest.mark.requirement("TRN-005")
def test_datasource_exposes_training_samples_via_task(tmp_path, evidence_output_dir):
    report = EvidenceReport(subject="Coronary dataset training interface")

    ds = _make_datasource(tmp_path)
    ds.create_partitions(DeterministicHoldoutSplit(train=0.7, val=0.15, seed=0))

    task = CoronaryCalciumTask()
    train_ids = ds.get_train_ids()

    sample_count = 0
    for pid in train_ids[:2]:
        patient_sample = ds.get_patient(pid)
        for sample in task.generate_training_samples(patient_sample):
            if "input" not in sample or "target" not in sample:
                report.error(f"Sample from patient {pid} missing 'input' or 'target'", "TRN-005")
            sample_count += 1

    if sample_count == 0:
        report.error("No training samples produced from the datasource", "TRN-005")

    report.auto_save("TRN005_dataset_training_interface", evidence_output_dir)
    assert not report.has_errors, report.summary()
