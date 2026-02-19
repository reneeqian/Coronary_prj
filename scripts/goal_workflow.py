medimgds = MedicalImageDataSource(datasetpath, ingestor)
imshow(medimgds.get_slice(patient_index=0, slice_index=0))
volshow(medimgds.get_volume(patient_index=0))
print(medimgsds.get_num_patients())
train_ds, val_ds, test_ds = medimgds.get_train_val_test_split(split_strategy=DeterministicHoldoutSplitStrategy())
trainer = MedicalImageTrainer(train_ds, val_ds, test_ds, model, training_config)
results = trainer.train()
# results object contains training history, best model checkpoint path, and final evaluation metrics on the test set.  it also contains directory info for where logs and artifacts are saved.  