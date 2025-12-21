# Makefile for the project

.PHONY: help install train eval multirun clean tensorboard list-ckpts

# Default command
help:
	@echo "Project Makefile - Available commands"
	@echo ""
	@echo "Basic Commands:"
	@echo "  install      - Install dependencies"
	@echo "  train        - Run single training"
	@echo "  eval         - Run evaluation on checkpoint"
	@echo "  multirun     - Train multiple models sequentially"
	@echo "  tensorboard  - Start TensorBoard server"
	@echo "  list-ckpts   - List all available checkpoints"
	@echo "  clean        - Clean up generated files"
	@echo ""
	@echo "Training Examples:"
	@echo "  make train                                    # Train with defaults"
	@echo "  make train ARGS='model=resnet18'              # Train ResNet18"
	@echo "  make train ARGS='experiment_name=my_exp'      # Named experiment"
	@echo "  make train ARGS='trainer.max_epochs=50'       # Override epochs"
	@echo "  make train ARGS='data.batch_size=128'         # Override batch size"
	@echo ""
	@echo "Multirun Examples:"
	@echo "  make multirun MODELS='simple_cnn,resnet18,resnet50'"
	@echo "  make multirun MODELS='simple_cnn,resnet18' ARGS='trainer.max_epochs=20'"
	@echo ""
	@echo "Evaluation Examples:"
	@echo "  make eval CKPT='logs/.../checkpoints/best.ckpt'"
	@echo "  make eval CKPT='logs/.../checkpoints/best.ckpt' ARGS='model=resnet18'"

install:
	pip install -r requirements.txt

# Train - use ARGS variable for Hydra overrides
train:
	python3 src/train.py $(ARGS)

# Eval - requires CKPT parameter
eval:
	@if [ -z "$(CKPT)" ]; then \
		echo "ERROR: Checkpoint path required. Use: make eval CKPT=/path/to/checkpoint.ckpt"; \
		exit 1; \
	fi
	python3 src/eval.py ckpt_path=$(CKPT) $(ARGS)

# Multirun - train multiple models
multirun:
	@if [ -z "$(MODELS)" ]; then \
		echo "Training default models: simple_cnn, resnet18, resnet50"; \
		python3 src/train.py -m model=simple_cnn,resnet18,resnet50 $(ARGS); \
	else \
		echo "Training models: $(MODELS)"; \
		python3 src/train.py -m model=$(MODELS) $(ARGS); \
	fi

# Start TensorBoard
tensorboard:
	@echo "Starting TensorBoard..."
	@echo "View at: http://localhost:6006"
	tensorboard --logdir=./logs

# List all checkpoints
list-ckpts:
	@if [ -f scripts/list_checkpoints.py ]; then \
		python3 scripts/list_checkpoints.py; \
	else \
		find logs -name "*.ckpt" -type f; \
	fi

# Clean up
clean:
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf logs/* outputs/* multirun/*
	@echo "Cleaned up generated files."