# MLOps Project Documentation

## Overview
This is a production-ready MLOps project for MNIST handwritten digit recognition with:
- End-to-end training pipeline
- Model serving via FastAPI
- Data and model monitoring
- Automated retraining capabilities
- Docker containerization

## Project Structure

```
Handwritten-Digit-Reconigtion/
│
├── .gitignore                          # Git ignore rules
├── .env.example                        # Environment variables template
├── requirements.txt                    # Production dependencies
├── requirements-dev.txt                # Development dependencies
├── Dockerfile                          # Container configuration
├── docker-compose.yml                  # Multi-container orchestration
├── Makefile                            # Common tasks automation
├── README.md                           # Project documentation
├── LICENSE
│
├── config/                             # Configuration management
│   ├── __init__.py
│   ├── config.yaml                    # Main configuration file
│   ├── training_config.yaml           # Training hyperparameters
│   ├── deployment_config.yaml         # Deployment settings
│   ├── monitoring_config.yaml         # Monitoring thresholds
│   └── logging_config.py              # Logging configuration
│
├── data/                               # Data management (gitignored)
│   ├── raw/                           # Original raw data
│   │   └── mnist/
│   │       ├── train-images.idx3-ubyte
│   │       ├── train-labels.idx1-ubyte
│   │       ├── t10k-images.idx3-ubyte
│   │       └── t10k-labels.idx1-ubyte
│   │
│   ├── processed/                     # Cleaned and preprocessed data
│   │   ├── train_features.pkl
│   │   ├── train_labels.pkl
│   │   ├── test_features.pkl
│   │   └── test_labels.pkl
│   │
│   ├── external/                      # External datasets
│   └── validation/                    # Validation datasets for drift detection
│
├── models/                             # Model management
│   ├── saved/                         # Trained model checkpoints (gitignored)
│   │   ├── v1/
│   │   │   ├── model.pt               # PyTorch model
│   │   │   ├── metadata.json          # Model metadata
│   │   │   ├── metrics.json           # Performance metrics
│   │   │   └── hyperparams.yaml       # Training hyperparameters
│   │   └── latest/
│   │
│   ├── artifacts/                     # MLflow artifacts (gitignored)
│   └── README.md                      # Model versioning documentation
│
├── src/                                # Source code (main module)
│   ├── __init__.py
│   │
│   ├── data/                          # Data ingestion and processing
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Data loading utilities
│   │   ├── data_preprocessor.py      # Data preprocessing logic
│   │   ├── data_validator.py         # Data validation rules
│   │   └── augmentation.py           # Data augmentation techniques
│   │
│   ├── models/                        # Custom model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py             # Abstract base class for models
│   │   ├── neural_network.py         # Custom CNN/DNN from scratch
│   │   ├── layers.py                 # Custom layer implementations
│   │   └── losses.py                 # Custom loss functions
│   │
│   ├── training/                      # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main training loop
│   │   ├── optimizer.py              # Custom optimizer implementations
│   │   ├── callbacks.py              # Training callbacks (early stopping, etc)
│   │   └── metrics.py                # Custom metrics calculation
│   │
│   ├── evaluation/                    # Model evaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py              # Evaluation framework
│   │   ├── metrics_calculator.py      # Metrics: accuracy, F1, confusion matrix
│   │   └── visualizer.py             # Result visualization
│   │
│   ├── inference/                     # Model serving & inference
│   │   ├── __init__.py
│   │   ├── predictor.py              # Prediction logic
│   │   ├── model_loader.py           # Model loading utilities
│   │   └── preprocessor.py           # Input preprocessing for inference
│   │
│   ├── monitoring/                    # Monitoring and drift detection
│   │   ├── __init__.py
│   │   ├── data_drift_detector.py    # Detect data distribution shifts
│   │   ├── model_drift_detector.py   # Detect model performance degradation
│   │   ├── metrics_tracker.py        # Track performance over time
│   │   └── alerter.py                # Alert system
│   │
│   ├── api/                           # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   ├── routes.py                 # API endpoints
│   │   ├── schemas.py                # Request/response schemas (Pydantic)
│   │   ├── middleware.py             # Custom middleware
│   │   └── dependencies.py           # Dependency injection
│   │
│   ├── pipelines/                     # Orchestration pipelines
│   │   ├── __init__.py
│   │   ├── data_pipeline.py          # Data ingestion pipeline
│   │   ├── training.py               # End-to-end training pipeline
│   │   ├── evaluation.py             # Evaluation pipeline
│   │   ├── deployment.py             # Deployment pipeline
│   │   ├── monitoring.py             # Monitoring pipeline
│   │   └── retraining.py             # Automated retraining logic
│   │
│   ├── utils/                         # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py                 # Logging utilities
│   │   ├── config_handler.py         # Config file handling
│   │   ├── file_utils.py             # File I/O utilities
│   │   ├── metrics_utils.py          # Metrics calculation helpers
│   │   └── device_utils.py           # GPU/CPU device management
│   │
│   └── exceptions/                    # Custom exceptions
│       ├── __init__.py
│       ├── data_exceptions.py        # Data-related errors
│       ├── model_exceptions.py       # Model-related errors
│       └── api_exceptions.py         # API-related errors
│
├── notebooks/                          # Jupyter notebooks for EDA & experiments
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_data_preprocessing.ipynb    # Data cleaning & preprocessing
│   ├── 03_model_experimentation.ipynb # Model architecture exploration
│   ├── 04_hyperparameter_tuning.ipynb # HPO experiments
│   ├── 05_model_analysis.ipynb       # Model interpretation & visualization
│   └── README.md                     # Notebook guidelines
│
├── tests/                             # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py                   # Pytest configuration & fixtures
│   ├── test_data/
│   │   ├── __init__.py
│   │   ├── test_data_loader.py       # Data loading tests
│   │   ├── test_preprocessor.py      # Preprocessing tests
│   │   └── test_validator.py         # Validation tests
│   ├── test_models/
│   │   ├── __init__.py
│   │   ├── test_neural_network.py    # Model architecture tests
│   │   └── test_layers.py            # Layer tests
│   ├── test_training/
│   │   ├── __init__.py
│   │   └── test_trainer.py           # Training logic tests
│   ├── test_inference/
│   │   ├── __init__.py
│   │   └── test_predictor.py         # Prediction tests
│   ├── test_api/
│   │   ├── __init__.py
│   │   └── test_routes.py            # API endpoint tests
│   └── test_monitoring/
│       ├── __init__.py
│       └── test_drift_detector.py    # Drift detection tests
│
├── logs/                              # Application logs (gitignored)
│   ├── training.log
│   ├── inference.log
│   ├── monitoring.log
│   ├── errors.log
│   └── api.log
│
├── scripts/                           # Standalone scripts
│   ├── download_data.py              # Download MNIST dataset
│   ├── train_model.py                # Training entry point
│   ├── evaluate_model.py             # Evaluation entry point
│   ├── check_drift.py                # Drift detection script
│   ├── retrain_pipeline.sh           # Retraining automation (cron job)
│   └── setup_environment.sh          # Environment setup script
│
├── docs/                              # Documentation
│   ├── README.md                     # Main documentation
│   ├── architecture.md               # System architecture
│   ├── data_flow.md                  # Data pipeline documentation
│   ├── model_training.md             # Training process documentation
│   ├── deployment.md                 # Deployment guide
│   ├── monitoring.md                 # Monitoring setup
│   ├── api_docs.md                   # API documentation
│   └── contributing.md               # Contribution guidelines
│
├── .github/                           # GitHub-specific files
│   ├── workflows/
│   │   ├── ci.yml                    # CI/CD pipeline
│   │   ├── training.yml              # Automated training workflow
│   │   ├── deployment.yml            # Deployment workflow
│   │   └── monitoring.yml            # Monitoring workflow
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
└── mlruns/                            # MLflow experiment tracking (gitignored)
    └── [auto-generated by MLflow]


```

## Quick Start for user

### 1. Setup Environment
```bash
make install
cp .env.example .env
```

### 2. Download Data
```bash
bash scripts/download_data.sh
```

### 3. Train Model
```bash
make train
```

### 4. Run API Server
```bash
uvicorn src.api.main:app --reload
```

### 5. Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [[0.0, ...], ...]}'
```

## Testing
```bash
make test
```

## Docker Deployment
```bash
docker-compose up -d
```

## Monitoring
```bash
bash scripts/monitor_system.sh
```

## Key Features

✅ **Custom Model**: Built CNN from scratch without pretrained models
✅ **Training Pipeline**: Full training loop with validation and checkpointing
✅ **Monitoring**: Data drift and model drift detection
✅ **API**: FastAPI server for predictions
✅ **CI/CD Ready**: Docker & GitHub Actions support
✅ **Testing**: Comprehensive unit and integration tests
✅ **Logging**: Structured logging across all modules
✅ **Configuration**: YAML-based config management

## Performance

Expected accuracy: ~99% on MNIST test set

## Author


## License
MIT License
