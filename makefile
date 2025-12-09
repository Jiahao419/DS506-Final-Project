VENV_NAME := cs506_venv
PYTHON := python
REQ_FILE := requirements.txt

# Default target
all: run

# Create virtual environment
$(VENV_NAME)/Scripts/activate:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV_NAME)

# Install dependencies
install: $(VENV_NAME)/Scripts/activate
	@echo "Installing dependencies..."
	$(VENV_NAME)/Scripts/pip install -r $(REQ_FILE)
	@echo "Finished..."

# Run code in sequence
run: install
	@echo "Running $(SCRIPT)..."
	$(VENV_NAME)/Scripts/python src/put actual source filename here.py
# Copy line above for all other .py files needed to run. Runs from top to bottom, order important.
	@echo "Finished. Check outputs for results."

# Clean virtual environment
clean:
ifeq ($(OS),Windows_NT)
	@if exist "$(VENV_NAME)" rmdir /S /Q "$(VENV_NAME)"
	@echo "Virtual Environment $(VENV_NAME) deleted..."
else
	@rm -rf "$(VENV_NAME)"
	@echo "Virtual Environment $(VENV_NAME) deleted..."
endif