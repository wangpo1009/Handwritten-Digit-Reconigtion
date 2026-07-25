# =========================================================
# Configuration
# =========================================================
PYTHON ?= python3.11
VENV := .venv
MONITOR_VENV := .venv-monitor
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV_PYTHON) -m pip
VENV_STAMP := $(VENV)/.requirements-installed
MONITOR_PYTHON := $(MONITOR_VENV)/bin/python
MONITOR_PIP := $(MONITOR_PYTHON) -m pip
MONITOR_STAMP := $(MONITOR_VENV)/.requirements-installed
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

.PHONY: help setup setup-monitor setup-all install install-monitor install-dev \
	format lint test test-monitor train evaluate notebook api monitor \
	shell monitor-shell retrain clean clean-venvs

# =========================================================
# Commands
# =========================================================

help: ## Hiển thị các lệnh có sẵn
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip

$(MONITOR_PYTHON):
	$(PYTHON) -m venv $(MONITOR_VENV)
	$(MONITOR_PIP) install --upgrade pip

$(VENV_STAMP): requirements.txt $(VENV_PYTHON)
	$(VENV_PIP) install -r requirements.txt
	@touch $(VENV_STAMP)

$(MONITOR_STAMP): requirements-monitor.txt $(MONITOR_PYTHON)
	$(MONITOR_PIP) install -r requirements-monitor.txt
	@touch $(MONITOR_STAMP)

setup: $(VENV_STAMP) ## Tạo .venv và cài môi trường chính

setup-monitor: $(MONITOR_STAMP) ## Tạo .venv-monitor và cài môi trường monitoring

setup-all: setup setup-monitor ## Tạo và cài môi trường setup và monitor

install: setup ## Alias của setup

install-monitor: setup-monitor ## Alias của setup-monitor

install-dev: $(VENV_PYTHON) ## Cài dependencies phát triển vào .venv
	$(VENV_PIP) install -r requirements-dev.txt

format: ## Tự động định dạng code (Black, Isort)
	$(VENV_PYTHON) -m black src/ tests/
	$(VENV_PYTHON) -m isort src/ tests/

lint: ## Kiểm tra chất lượng code và kiểu dữ liệu
	$(VENV_PYTHON) -m ruff check src/ tests/
	$(VENV_PYTHON) -m mypy src/

test: ## Chạy Unit tests và xuất báo cáo độ bao phủ (Coverage)
	$(VENV_PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-monitor: setup-monitor ## Chạy riêng các test monitoring bằng .venv-monitor
	$(MONITOR_PYTHON) -m pytest tests/test_monitoring -v

train: ## Huấn luyện mô hình (Dùng biến môi trường từ .env)
	$(VENV_PYTHON) -m src.pipelines.training

evaluate: ## Đánh giá mô hình trên tập Test
	$(VENV_PYTHON) -m src.pipelines.evaluation

notebook: setup ## Khởi động Jupyter bằng kernel của .venv
	$(VENV_PYTHON) -m jupyter notebook

api: setup ## Khởi động FastAPI bằng môi trường chính
	$(VENV_PYTHON) -m uvicorn src.api.main:app --reload

monitor: setup-monitor ## Chạy pipeline Data Drift bằng môi trường monitoring
	$(MONITOR_PYTHON) -m src.pipelines.monitoring

shell: setup ## Mở shell đã kích hoạt .venv
	@echo "Đang mở shell của $(VENV). Gõ 'exit' để thoát."
	@exec $(SHELL) -c '. $(VENV)/bin/activate && exec $(SHELL) -i'

monitor-shell: setup-monitor ## Mở shell đã kích hoạt .venv-monitor
	@echo "Đang mở shell của $(MONITOR_VENV). Gõ 'exit' để thoát."
	@exec $(SHELL) -c '. $(MONITOR_VENV)/bin/activate && exec $(SHELL) -i'

retrain: ## Chạy pipeline tái huấn luyện tự động
	$(VENV_PYTHON) -m src.pipelines.retraining

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

clean-venvs: ## Xóa cả hai virtual environment
	$(RM) $(VENV) $(MONITOR_VENV)
