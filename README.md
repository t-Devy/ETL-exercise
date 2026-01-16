> [!NOTE]
> This project was built via "vibe coding" as a foundational architecture exercise, prioritizing correctness, readability. 
> The intention is to familiarize with program architecture in its various forms to build intuition and independent programming capabilities, without the help of IDE suggestion tools.
> All functionality is operational and verifiable via saved artifacts.

# ETL Exercise — Foundational ML Data Pipeline

This repository demonstrates an abstract, one file ETL (Extract–Transform–Load), and a modular ETL pipeline (multiple classes and files), designed to mirror real-world machine learning data workflows.
The focus is on clarity, architectural separation, and artifact-driven validation.

## Project Goals

- Practice end-to-end ML data architecture at a foundational level
- Translate conceptual ETL understanding into clean, fluent Python code
- Mirror industry-style project structure (paths, artifacts, modular steps)
- Produce verifiable outputs at every stage (no black boxes)

## Dataset

The mock dataset represents College & Career Readiness indicators, including:

`Academic performance` (GPA, SAT scores)

`Engagement metrics` (attendance, CTE hours)

`Experiential learning` (internships)

`Psychosocial variables` (self-efficacy, clarity of career interests)

The target variable is a synthetic `readiness score`, suitable for regression modeling.

# Pipeline Overview

1. Extract (extract_class_vibe.py)
- Loads raw CSV data
- Returns a pandas DataFrame
- Saves a serialized artifact (ccr_raw.pkl)
- python -m src.extract_class_vibe

### Artifact produced:

`data/ccr_raw.pkl`

## 2. Transform (transform_class_vibe.py)

- Loads extracted artifact
- Selects feature and target columns
- Splits into train/validation sets
- Scales features using StandardScaler
- Persists all intermediate outputs
- python -m src.transform_class_vibe


### Artifacts produced:

NumPy arrays (`X_train`, `X_val`, `y_train`, `y_val`)

Fitted scaler (`scaler.pkl`)

Feature schema (`feature_columns.json`)

## 3. Load (load_class_vibe.py)

- Loads transformed arrays
- Wraps them in a custom PyTorch Dataset
- Builds train/validation DataLoaders
- Saves loader metadata for inspection
- python -m src.load_class_vibe


### Artifacts produced:

`train_loader_info.json`
`val_loader_info.json`

These confirm batch sizes, shapes, and loader configuration without running a model.

# Why This Project Matters

This project intentionally avoids shortcuts like notebooks or monolithic scripts.
Instead, it emphasizes:

- Separation of concerns (Extract vs Transform vs Load)
- Artifact-based validation (inspect outputs at every step)
- Runtime-safe path management via a centralized paths.py
- Production-aligned thinking

## This is the same mental model used in:

ML platform teams

Feature pipelines

Model training infrastructure

Next Natural Extensions (Not Implemented Here)

Training loop (train.py)

Model checkpointing

Inference pipeline

Experiment tracking

Dataset versioning

# How to Run End-to-End

From the project root:

python -m src.extract_class_vibe

python -m src.transform_class_vibe

python -m src.load_class_vibe


All artifacts will be generated automatically if directories do not exist.






