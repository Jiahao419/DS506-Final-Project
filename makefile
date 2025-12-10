VENV_NAME := cs506_venv
REQ_FILE  := requirements.txt

# Detect OS-specific venv bin folder
ifeq ($(OS),Windows_NT)
	BIN_DIR := $(VENV_NAME)/Scripts
else
	BIN_DIR := $(VENV_NAME)/bin
endif

PYTHON := $(BIN_DIR)/python
PIP    := $(BIN_DIR)/pip

.PHONY: all setup install run clean data weather train-baseline train-weather time-split per-group eda

all: run

# 1. Create virtual environment
setup:
	@echo "Creating virtual environment $(VENV_NAME)..."
	python -m venv $(VENV_NAME)

# 2. Install dependencies
install: setup
	@echo "Installing dependencies from $(REQ_FILE)..."
	$(PIP) install -r $(REQ_FILE)
	@echo "Finished installing dependencies."

# 3. Individual pipeline steps (modify filenames as needed)
data: install
	@echo "Running data cleaning..."
	$(PYTHON) src/data_cleaning.py

weather: install
	@echo "Downloading & merging weather..."
	$(PYTHON) src/weather_download.py
	$(PYTHON) src/weather_merge.py

train-baseline: install
	@echo "Training baseline models..."
	$(PYTHON) src/model_training.py

train-weather: install
	@echo "Training weather + engineered models..."
	$(PYTHON) src/model_training_weather.py

time-split: install
	@echo "Training time-split models..."
	$(PYTHON) src/model_training_time_split.py

per-group: install
	@echo "Computing per-group metrics..."
	$(PYTHON) src/per_group_metrics.py

eda: install
	@echo "Generating EDA plots..."
	$(PYTHON) src/eda_visualization.py

# Main entry point to reproduce the full experiment pipeline
run: install
	@echo "Running full pipeline..."
	$(PYTHON) src/data_cleaning.py
	$(PYTHON) src/weather_download.py
	$(PYTHON) src/weather_merge.py
	$(PYTHON) src/model_training.py
	$(PYTHON) src/model_training_weather.py
	$(PYTHON) src/model_training_time_split.py
	$(PYTHON) src/per_group_metrics.py
	@echo "Pipeline finished. Check outputs/ for results."

# Remove virtual environment
clean:
ifeq ($(OS),Windows_NT)
	@if exist "$(VENV_NAME)" rmdir /S /Q "$(VENV_NAME)"
else
	@rm -rf "$(VENV_NAME)"
endif
	@echo "Virtual environment $(VENV_NAME) deleted."
