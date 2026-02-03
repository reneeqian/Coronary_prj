from src.datasets.coronary_ct_dataset import CoronaryCTDataset

ds = CoronaryCTDataset("data/processed")

sample = ds[0]
print(sample["image"].shape)
print(sample["label"].shape)
print(sample["id"])
