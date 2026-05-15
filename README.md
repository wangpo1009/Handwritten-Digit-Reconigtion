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
├── src/                    # Source code (modules)
│   ├── data/              # Data loading & preprocessing
│   ├── models/            # Custom model implementations
│   ├── training/          # Training pipeline
│   ├── inference/         # Model serving
│   ├── monitoring/        # Drift detection & monitoring
│   ├── api/               # FastAPI application
│   ├── pipelines/         # End-to-end pipelines
│   ├── utils/             # Utility functions
│   └── exceptions/        # Custom exceptions
│
├── config/                # Configuration management
├── data/                  # Data directory
├── models/                # Model checkpoints
├── logs/                  # Application logs
├── tests/                 # Unit and integration tests
├── notebooks/             # EDA and experiments
├── scripts/               # Standalone scripts
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-container setup
├── Makefile               # Task automation
└── requirements.txt       # Python dependencies
```

## Quick Start

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
Your Name (@wangpo1009)

## License
MIT License
