# System Overview

## Purpose

This project demonstrates a deterministic AI training pipeline for medical image data designed with regulatory-grade software discipline.

It models how an ML system might be structured to support FDA Software as a Medical Device (SaMD) expectations while remaining lean and maintainable.

This is not a clinical product.

---

## System Boundary

The system includes:

- Medical image ingestion
- Dataset validation and deterministic access
- Train/validation/test splitting
- Model training
- Verification testing
- Evidence generation
- Requirements traceability

It excludes:
- Clinical decision support logic
- Real-world deployment infrastructure
- Regulatory submission artifacts

---

## Design Principles

- Deterministic dataset access
- Explicit validation contracts
- Traceable requirements
- Automated verification evidence
- Minimal manual documentation burden