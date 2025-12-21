# PyTorch Template with Hydra

A minimal, Lightning-inspired training framework built from scratch with Hydra configuration management.

## 🎯 Features

- ✅ Clean training loop with callbacks
- ✅ Model checkpointing (best + last)
- ✅ Early stopping
- ✅ Rich progress bars
- ✅ TensorBoard & CSV logging
- ✅ Hydra configuration management
- ✅ Easy architecture swapping
- ✅ Single GPU support
- ✅ Makefile for convenient command execution

## 📁 Project Structure

```
.
├── configs/
│   ├── train.yaml              # Main config
│   ├── data/
│   │   └── mnist.yaml          # Data configs
│   ├── model/
│   │   ├── simple_cnn.yaml     # Model configs
│   │   └── resnet18.yaml
│   ├── trainer/
│   │   └── default.yaml        # Trainer configs
│   ├── callbacks/
│   │   └── default.yaml        # Callback configs
│   ├── logger/
│   │   ├── tensorboard.yaml    # Logger configs
│   │   └── csv.yaml
│   └── paths/
│       └── default.yaml        # Path configs
├── src/
│   ├── core/
│   │   ├── trainer.py          # Main Trainer class
│   │   ├── callbacks.py        # Callbacks
│   │   └── loggers.py          # Loggers
│   ├── data/
│   │   ├── datamodule.py       # Base DataModule
│   │   └── mnist_datamodule.py # MNIST implementation
│   ├── models/
│   │   ├── base_module.py      # Base model class
│   │   ├── mnist_classifier.py # MNIST classifier
│   │   └── components/
│   │       ├── simple_cnn.py   # Simple CNN
│   │       └── mnist_resnet.py # ResNet variants
│   └── train.py                # Training script
├── Makefile                     # Convenient command shortcuts
├── data/                        # Downloaded datasets
├── logs/                        # Training logs & checkpoints
└── outputs/                     # Hydra outputs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
make install
# Or: pip install -r requirements.txt
```

### 2. Train with Default Config

**Using Makefile:**
```bash
make train
```

**Direct Python:**
```bash
python src/train.py
```

This will:
- Train SimpleCNN on MNIST
- Save logs to `logs/train/simple_cnn/{experiment_name}/{timestamp}/`
- Save checkpoints to `logs/train/simple_cnn/{experiment_name}/{timestamp}/checkpoints/`
- Log metrics to TensorBoard
- Show rich progress bars

## 📋 Makefile Commands

The Makefile provides convenient shortcuts for common tasks. View all available commands:

```bash
make help
```

### Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `make install` | Install dependencies | `make install` |
| `make train` | Run training | `make train` |
| `make eval` | Evaluate checkpoint | `make eval CKPT=path/to/checkpoint.ckpt` |
| `make multirun` | Train multiple models | `make multirun MODELS='simple_cnn,resnet18'` |
| `make tensorboard` | Start TensorBoard | `make tensorboard` |
| `make list-ckpts` | List all checkpoints | `make list-ckpts` |
| `make clean` | Clean generated files | `make clean` |

### Training with Makefile

**Basic training:**
```bash
make train
```

**With custom arguments (use ARGS variable):**
```bash
# Train with ResNet18
make train ARGS='model=resnet18'

# Named experiment
make train ARGS='experiment_name=my_experiment'

# Override multiple parameters
make train ARGS='model=resnet18 trainer.max_epochs=50'

# Change batch size
make train ARGS='data.batch_size=128'

# Combine multiple overrides
make train ARGS='experiment_name=test_run model=resnet18 trainer.max_epochs=20 data.batch_size=64'
```

### Multirun with Makefile

Train multiple models sequentially:

```bash
# Default models (simple_cnn, resnet18, resnet50)
make multirun

# Specify models
make multirun MODELS='simple_cnn,resnet18'

# With additional arguments
make multirun MODELS='simple_cnn,resnet18' ARGS='trainer.max_epochs=20'

# Complex example
make multirun MODELS='simple_cnn,resnet18,resnet50' ARGS='data.batch_size=128 trainer.max_epochs=30'
```

### Evaluation with Makefile

```bash
# Evaluate a checkpoint
make eval CKPT='logs/train/simple_cnn/my_exp/2024-12-05_14-30-00/checkpoints/best.ckpt'

# Evaluate with specific model config
make eval CKPT='logs/.../checkpoints/best.ckpt' ARGS='model=resnet18'
```

### TensorBoard with Makefile

```bash
# Start TensorBoard server
make tensorboard

# Then open http://localhost:6006 in your browser
```

## 🎛️ Direct Python Usage

You can also use Python directly without the Makefile:

### Basic Training

```bash
# Default config
python src/train.py

# Change model
python src/train.py model=resnet18

# Named experiment
python src/train.py experiment_name=my_experiment

# Multiple overrides
python src/train.py model=resnet18 trainer.max_epochs=50 data.batch_size=128
```

### Multirun (Multiple Experiments)

```bash
# Compare architectures
python src/train.py -m model=simple_cnn,resnet18,resnet50

# Grid search over hyperparameters
python src/train.py -m model=simple_cnn,resnet18 data.batch_size=64,128

# Complex multirun
python src/train.py -m model=simple_cnn,resnet18 trainer.max_epochs=20,50 data.batch_size=64,128
```

### Advanced Overrides

```bash
# Change optimizer learning rate
python src/train.py model.optimizer.lr=0.0001

# Disable early stopping
python src/train.py callbacks.early_stopping=null

# Change early stopping patience
python src/train.py callbacks.early_stopping.patience=10

# Change checkpoint monitor
python src/train.py callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"

# Resume from checkpoint
python src/train.py ckpt_path=logs/.../checkpoints/last.ckpt

# Test only (no training)
python src/train.py train=false test=true
```

## 📂 Output Structure

After training, your files will be organized as:

```
logs/
└── train/
    └── simple_cnn/              # Model name
        └── my_experiment/       # Experiment name (or timestamp)
            └── 2024-12-05_14-30-00/  # Run timestamp
                ├── checkpoints/
                │   ├── best.ckpt
                │   └── last.ckpt
                ├── events.out.tfevents.*  # TensorBoard logs
                └── hparams.txt

outputs/                         # Hydra's own outputs
└── 2024-12-05_14-30-00/
    └── .hydra/
        └── config.yaml          # Full resolved config
```

## 📝 Creating New Models

### Step 1: Create Model Component

Create `src/models/components/my_model.py`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Your architecture here
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc = nn.Linear(32 * 26 * 26, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
```

### Step 2: Create Config

Create `configs/model/my_model.yaml`:

```yaml
_target_: src.models.mnist_classifier.MNISTClassifier

name: my_model  # Add this for organized logging

net:
  _target_: src.models.components.my_model.MyModel
  num_classes: 10

optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001

scheduler: null  # Optional
```

### Step 3: Train

```bash
# Using Makefile
make train ARGS='model=my_model'

# Direct Python
python src/train.py model=my_model
```

## 🎓 Understanding the Components

### BaseModule

- Like `LightningModule` but simpler
- Inherit and implement: `forward()`, `training_step()`, `validation_step()`, `configure_optimizers()`

### DataModule

- Organizes data loading
- Implement: `setup()`, then use `train_dataloader()`, `val_dataloader()`, `test_dataloader()`

### Trainer

- Main training loop
- Handles: epochs, batches, validation, device management
- Calls callbacks at appropriate times

### Callbacks

- Hook into training process
- Available: `ModelCheckpoint`, `EarlyStopping`, `RichProgressBar`
- Easy to create custom callbacks!

### Loggers

- Track experiments
- Available: `TensorBoardLogger`, `CSVLogger`

## 🔧 Advanced Usage

### Resume Training

```bash
# Using Makefile
make train ARGS='ckpt_path=logs/.../checkpoints/last.ckpt'

# Direct Python
python src/train.py ckpt_path=logs/.../checkpoints/last.ckpt
```

### Test Only

```bash
# Using Makefile
make train ARGS='train=false test=true'

# Direct Python
python src/train.py train=false test=true
```

### Change Logger

```bash
# Use CSV logger instead
python src/train.py logger=csv

# Use both TensorBoard and CSV
python src/train.py logger=tensorboard,csv
```

### Custom Callbacks

```bash
# Disable early stopping
python src/train.py callbacks.early_stopping=null

# Change patience
python src/train.py callbacks.early_stopping.patience=10

# Monitor different metric
python src/train.py callbacks.model_checkpoint.monitor="val/loss" callbacks.model_checkpoint.mode="min"
```

### Custom Schedulers

Modify `configs/model/your_model.yaml`:

```yaml
scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  _partial_: true
  step_size: 5
  gamma: 0.1
```

Or override via CLI:

```bash
python src/train.py model.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR
```

## 🆚 vs PyTorch Lightning

### What's Different?

- ❌ No multi-GPU support
- ❌ No gradient accumulation
- ❌ No mixed precision
- ✅ Much simpler to understand
- ✅ ~500 lines of code vs thousands
- ✅ Same clean API

### When to Use This?

- Learning how training frameworks work
- Single GPU experiments
- Don't need advanced features
- Want full control

### When to Use Lightning?

- Production use
- Multi-GPU training
- Advanced features (profiling, pruning, etc.)
- Stable, battle-tested code

## 📊 Example Workflows

### Experiment 1: Compare Architectures

```bash
# Using Makefile
make multirun MODELS='simple_cnn,resnet18,resnet50'

# Direct Python
python src/train.py -m model=simple_cnn,resnet18,resnet50
```

Results saved to separate directories per model. Compare in TensorBoard!

### Experiment 2: Hyperparameter Search

```bash
# Grid search over learning rates and batch sizes
python src/train.py -m model.optimizer.lr=0.001,0.0001,0.00001 data.batch_size=32,64,128
```

Creates 9 experiments (3 LRs × 3 batch sizes)!

### Experiment 3: Quick Ablation Study

```bash
# Test with/without early stopping
python src/train.py -m callbacks.early_stopping.patience=3,5,10

# Test different optimizers
python src/train.py -m model.optimizer._target_=torch.optim.Adam,torch.optim.SGD
```

### Experiment 4: Production Training

```bash
# Named experiment with specific hyperparameters
make train ARGS='experiment_name=production_v1 model=resnet18 trainer.max_epochs=100 data.batch_size=256'

# View results
make tensorboard
```

## 🐛 Troubleshooting

### PyTorch 2.6 Checkpoint Loading Error

Already handled! The `train.py` patches `torch.load` to use `weights_only=False`.

### CUDA Out of Memory

```bash
# Using Makefile
make train ARGS='data.batch_size=32'

# Direct Python
python src/train.py data.batch_size=32
```

### Validation Not Running

```bash
python src/train.py trainer.check_val_every_n_epoch=1
```

### Can't Find Checkpoints

```bash
# List all available checkpoints
make list-ckpts

# Or manually search
find logs -name "*.ckpt" -type f
```

### Arguments Not Working with Makefile

Make sure to use the `ARGS` variable:

```bash
# ✅ Correct
make train ARGS='model=resnet18'

# ❌ Wrong
make train model=resnet18
```

## 🧹 Cleanup

```bash
# Clean up all generated files
make clean

# Manual cleanup
rm -rf logs/* outputs/* multirun/*
```

## 📚 Learn More

- **Hydra Docs**: <https://hydra.cc/>
- **PyTorch Docs**: <https://pytorch.org/docs/>
- **Original Inspiration**: <https://github.com/nathanpainchaud/lightning-hydra-template>

## 💡 Pro Tips

- Use `experiment_name` to organize related runs
- Always check TensorBoard to compare experiments
- Use multirun (`-m`) for systematic comparisons
- Save important checkpoints with descriptive experiment names
- The `ARGS` variable in Makefile can take multiple space-separated overrides
- Use `make help` to see all available commands
- Logs are organized hierarchically: model → experiment → timestamp

Happy training! 🚀