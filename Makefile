.PHONY: help install install-dev format lint test train evaluate deploy clean

help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make install-dev      - Install dev dependencies"
	@echo "  make format           - Format code with black and isort"
	@echo "  make lint             - Run code quality checks"
	@echo "  make test             - Run unit tests"
	@echo "  make train            - Train the model"
	@echo "  make evaluate         - Evaluate model on test set"
	@echo "  make deploy           - Build and run Docker container"
	@echo "  make clean            - Clean up generated files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

format:
	black src/ tests/ notebooks/
	isort src/ tests/ notebooks/

lint:
	flake8 src/ tests/
	pylint src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python -m src.pipelines.training

evaluate:
	python -m src.pipelines.evaluation

monitor:
	python -m src.monitoring.data_drift

retrain:
	python -m src.pipelines.retraining

docker-build:
	docker build -t handwritten-digit-recognition:latest .

docker-run:
	docker run -p 8000:8000 \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/models/saved:/app/models/saved \
		handwritten-digit-recognition:latest

docker-push:
	docker tag handwritten-digit-recognition:latest $(DOCKER_REGISTRY)/handwritten-digit-recognition:latest
	docker push $(DOCKER_REGISTRY)/handwritten-digit-recognition:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf mlruns/ .mlflow/
