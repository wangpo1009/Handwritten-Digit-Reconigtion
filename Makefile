# =========================================================
# Configuration
# =========================================================
PYTHON := python
PIP := uv pip
APP_NAME := handwritten-digit-recognition
DOCKER_REGISTRY := your-registry-name# Thay bằng registry của nhóm

# Kiểm tra hệ điều hành để dùng lệnh xóa file phù hợp
ifeq ($(OS),Windows_NT)
    RM := rmdir /s /q
    DELETE := del /f /q
    PWD_COMMAND := ${CURDIR}
else
    RM := rm -rf
    DELETE := rm -f
    PWD_COMMAND := $(shell pwd)
endif

.PHONY: help install install-dev format lint test train evaluate deploy clean monitor retrain

# =========================================================
# Commands
# =========================================================

help: ## Hiển thị các lệnh có sẵn
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Cài đặt dependencies (Production)
	$(PIP) install -r requirements.txt

install-dev: ## Cài đặt dependencies cho phát triển và kiểm thử
	$(PIP) install -r requirements-dev.txt

format: ## Tự động định dạng code (Black, Isort)
	black src/ tests/ notebooks/
	isort src/ tests/ notebooks/

lint: ## Kiểm tra chất lượng code và kiểu dữ liệu
	ruff check src/ tests/
	mypy src/

test: ## Chạy Unit tests và xuất báo cáo độ bao phủ (Coverage)
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

train: ## Huấn luyện mô hình (Dùng biến môi trường từ .env)
	$(PYTHON) -m src.pipelines.training

evaluate: ## Đánh giá mô hình trên tập Test
	$(PYTHON) -m src.pipelines.evaluation

monitor: ## Kiểm tra Data Drift (Evidently)
	$(PYTHON) -m src.monitoring.data_drift

retrain: ## Chạy pipeline tái huấn luyện tự động
	$(PYTHON) -m src.pipelines.retraining

# --- Docker Ops ---

docker-build: ## Build Docker Image
	docker build -t $(APP_NAME):latest .

docker-run: ## Chạy Docker Container với Volume mapping cho Logs và Models
	docker run -p 8000:8000 \
		--env-file .env \
		-v $(PWD_COMMAND)/logs:/app/logs \
		-v $(PWD_COMMAND)/models/saved:/app/models/saved \
		$(APP_NAME):latest

# --- Cleanup ---

clean: ## Dọn dẹp cache, log và các file rác
	@echo "Cleaning up..."
	$(RM) .pytest_cache .coverage htmlcov .mypy_cache build dist *.egg-info mlruns
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete