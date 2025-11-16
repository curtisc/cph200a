# CPH200 Project 2: Lung Cancer Detection & Risk Prediction

## Project Overview
Multi-stage deep learning project for lung cancer detection, localization, and risk prediction using NLST (National Lung Screening Trial) CT scans. Built on PyTorch Lightning with WandB experiment tracking.

**Progression:** PathMNIST toy dataset (Part 1) → NLST cancer detection (Part 2) → NLST risk prediction (Part 3)

## Architecture & Key Components

### Three-Layer Design Pattern
1. **Dataset Layer** (`src/dataset.py`): Lightning DataModules handle data loading, preprocessing, and splits
   - `PathMnist`: 28x28 toy pathology images for rapid prototyping
   - `NLST`: 256×256×200 downsampled CT volumes with cancer labels, bounding boxes, LungRads scores
   - `NLST_Dataset`: Custom dataset handling 3D volumes with TorchIO transforms and mask generation

2. **Model Layer** (`src/lightning.py`): Lightning Modules define architectures and training logic
   - `Classifer` base class: Handles training/val/test loops, metrics (accuracy, AUC), optimizer setup
   - Implemented models: `Linear`, `MLP`, `CNN`, `ResNet18`, `CNN3D`
   - `RiskModel` skeleton: Multi-horizon survival prediction (TODO implementation)

3. **Execution Layer** (`scripts/main.py`, `scripts/dispatcher.py`): Experiment management
   - `main.py`: CLI interface using Lightning's ArgumentParser for all hyperparameters
   - `dispatcher.py`: Grid search runner - spawns parallel experiments from JSON configs

### Critical Data Flow Details

**NLST Label Structure:**
- `y`: Binary cancer within max_followup years (default 6)
- `y_seq`: Array `[0,0,1,1,1,1]` = cancer detected between years 2-3
- `y_mask`: Array `[1,1,0,0,0,0]` = observed only up to 2 years (censoring)
- `time_at_event`: Years until cancer OR censoring

**Bounding Box → Mask Pipeline** (`NLST_Dataset.get_scaled_annotation_mask`):
- Converts fractional [0,1] bounding boxes to pixel-precise masks
- Handles sub-pixel overlap via weighted intersection calculation
- Returns (1, Z, H, W) masks aligned with volumes via TorchIO transforms

**Transfer Learning Strategy** (`ResNet18`):
- Backbone frozen initially → train classifier head only
- Call `model.unfreeze_backbone()` for fine-tuning phase
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Development Workflows

### Running Experiments
```bash
# Single experiment with specific config
python scripts/main.py --train --model_name cnn --dataset_name pathmnist \
  --cnn.hidden_dim 128 --cnn.use_bn true \
  --trainer.max_epochs 50 --pathmnist.batch_size 32

# Grid search from JSON config
cd scripts && python dispatcher.py --config_path grid_search.json --num_workers 4
```

### Model/Dataset Registration
Add new architectures to `NAME_TO_MODEL_CLASS` and `NAME_TO_DATASET_CLASS` dicts in `main.py`. Lightning CLI automatically discovers hyperparameters from `__init__` signatures.

### Distributed Training (Multi-GPU)
Critical environment variables pre-set in `main.py`:
```python
os.environ['NCCL_IB_DISABLE'] = '1'  # No InfiniBand
os.environ['NCCL_P2P_DISABLE'] = '1'  # Use socket communication
```
Trainer automatically uses DDP when `devices > 1`. DataLoaders have `shuffle=False` (Lightning's DistributedSampler handles it).

### Metrics & Logging
- **Epoch-level AUC**: Models store outputs in `self.training_outputs` lists, compute AUC in `on_*_epoch_end()` hooks
- **sync_dist=True**: Required for accurate multi-GPU metrics aggregation
- **WandB Integration**: Auto-enabled via `pl.loggers.WandbLogger` in main.py
- **C-Index** (`src/cindex.py`): IPCW-weighted concordance index for survival analysis (used in RiskModel)

## Project-Specific Conventions

### Why TorchIO Over Standard Augmentations
TorchIO ensures **spatial consistency** between volumes and masks - rotations/flips apply identically to both. Standard torchvision transforms would desync annotations.

### Class Imbalance Handling
Dataset provides `class_balance` flag (NotImplementedError - student TODO). Common approaches:
- Weighted sampling in DataLoader
- Loss weighting via `class_weights` in `Classifer.__init__`
- SMOTE/oversampling

### Patient-Level Splits
`use_stratified_group_split=True`: Ensures no patient leakage across train/val/test by grouping all exams from same patient. Manually rebalances splits if positive cases end up in zero.

### Data Storage Strategy
Compressed CT scans stored in `data/compressed/*.pt.z` on **NVMe local storage** for fast I/O. Loaded via `joblib.load(path + ".z")`. Full metadata in `data/nlst-metadata/full_nlst_google.json`.

## Common Gotchas

1. **get_xy() format handling**: Classifer.get_xy() handles both tuple format `(x, y)` from PathMNIST and dict format `{'x': ..., 'y_seq': ...}` from NLST
2. **Batch size constraints**: 3D volumes require small batch sizes (typically 1-4) due to memory. PathMNIST supports 32-1024.
3. **VOXEL_SPACING constant**: (0.703125, 0.703125, 2.5) mm - TorchIO resample transform uses this to normalize physical spacing
4. **Checkpoint monitoring**: `monitor_key` defaults to `val_loss` but should be `val_auc` (mode='max') for Part 2+
5. **RiskModel TODO areas**: Forward pass, loss computation, metrics logging all need implementation per Part 3 requirements

## Performance Targets (from README)
- Part 1: PathMNIST val_acc ≥ 99%
- Part 2.1: NLST 1-year AUC ≥ 0.80
- Part 2.2: NLST 1-year AUC ≥ 0.87 (with localization)
- Part 3: Image-based 6-year AUC ≥ 0.76

## Key Files for Understanding
- `src/dataset.py:90-350`: NLST setup(), label generation, stratified splits
- `src/lightning.py:10-160`: Classifer base class training loop
- `src/lightning.py:500-630`: ResNet18 transfer learning implementation
- `src/dataset.py:370-480`: Bounding box → mask conversion logic
