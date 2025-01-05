###################################################################
# Data collection
###################################################################
.PHONY: data_collection
data_collection:
	@echo "Collecting data..."
	python3 data_collection/main.py

###################################################################
# Data Processing
###################################################################
.PHONY: data_processing
data_processing:
	@echo "Processing data..."
	python3 data_processing/main.py

###################################################################
# Model Training
###################################################################
.PHONY: model_training
model_training:
	@echo "Training Model..."
	python3 model_training/main.py

###################################################################
# Model Validation
###################################################################
.PHONY: model_validation
model_validation:
	@echo "Validating Model..."
	python3 model_validation/main.py

###################################################################
# Run the pipeline in seccession based on the config selected
###################################################################
.PHONY: all
all:
	@echo "Running the pipeline..."
	python3 data_collection/main.py
	@echo "Data collection completed..."
	python3 data_processing/main.py
	@echo "Data processing completed..."
	python3 model_training/main.py
	@echo "Model training completed..."
	python3 model_validation/main.py
	@echo "Model validation completed..."

###################################################################
# Help
###################################################################
.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  data_collection: Collect data"
	@echo "  data_processing: Process data"
	@echo "  model_training: Train model"
	@echo "  model_validation: Validate model"
	@echo "  all: Run the pipeline"