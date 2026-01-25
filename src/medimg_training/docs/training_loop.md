# Training Loop Semantics

This module provides a generic training loop independent of task or modality.

## Trainer responsibilities

- Iterate over batches
- Forward pass through the model
- Compute loss
- Backpropagation and optimization
- Metric aggregation and reporting

## Batch contract

Each batch is expected to be a dictionary containing:
- `image`: torch.Tensor
- `target` (optional)
- `meta` (optional)

## Design intent

- The trainer does not understand clinical meaning
- Task-specific logic lives in:
  - loss functions
  - metrics
  - model heads
