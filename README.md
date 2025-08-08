## Project Overview

This is a medical image segmentation evaluation toolkit for multi-rater annotations. The project evaluates segmentation models by comparing predictions against multiple annotator ground truth labels using specialized metrics that account for inter-annotator variability.

## Core Components

### Main Evaluation Scripts
- `diverse_performance.py` - Calculates set-level similarity metrics (GED, Dice_max, Dice_match, Dice_soft)
- `personalized_performance.py` - Evaluates individual expert Dice scores
- `metrics_set.py` - Core metric computation functions (Dice, IoU, GED calculations)

### Utility Scripts
- `count_classes.py` - Analyzes unique class labels in NIfTI files
- `create_test_data.py` - Generates synthetic test predictions from ground truth

## Common Commands

### Install Dependencies
```bash
uv sync
```

### Automatic Configuration (Recommended)
```bash
# For diverse performance evaluation with auto-detection
python diverse_performance.py --pred_path /path/to/prediction.nii.gz --gt_path /path/to/annotations/ --auto_config

# For personalized performance evaluation with auto-detection
python personalized_performance.py --pred_path /path/to/prediction.nii.gz --gt_path /path/to/annotations/ --auto_config

# Auto-config works with both file paths and directory paths
python diverse_performance.py --pred_path /path/to/predictions/ --auto_config
```

### Manual Configuration
```bash
# Multi-class segmentation with manual parameters
python diverse_performance.py --pred_path /path/to/predictions --gt_path /path/to/ground_truth --multiclass --num_classes 4

# K-fold cross-validation
python personalized_performance.py --pred_path /path/to/base --kfold --num_folds 5

# Custom patterns and parameters
python diverse_performance.py --pred_path /path/to/data --pred_pattern custom_pred_ --gt_pattern custom_gt_ --multiclass --num_classes 6
```

### Utility Commands
```bash
# Count classes in label files
python count_classes.py

# Generate synthetic test data
python create_test_data.py
```

## File Naming Conventions and Auto-Configuration

### Automatic Pattern Detection
The `--auto_config` flag automatically detects file patterns and parameters:
- **Intelligent pattern inference**: Extracts patterns from actual filenames
- **Sparse class handling**: Properly handles non-consecutive class indices (e.g., [0, 2, 3])
- **Directory separation**: Supports predictions and annotations in different directories
- **Parameter extraction**: Auto-detects num_classes, multiclass mode, and background handling

### Supported File Structures
```
# Single prediction file with annotations in subdirectory
prediction.nii.gz
annotations_folder/
  ├── label_annot_1.nii.gz
  └── label_annot_5.nii.gz

# Traditional patterns (also auto-detected)
pred_s1.nii.gz, pred_s2.nii.gz
label_annot_1.nii.gz, label_annot_5.nii.gz

# Custom ID patterns (auto-detected)
ID_90ae3af3_ID_8d77fcb5d2.nii.gz
ID_90ae3af3_ID_8d77fcb5d2/
  ├── label_annot_1.nii.gz
  └── label_annot_5.nii.gz
```

### Manual Pattern Override
Use `--pred_pattern` and `--gt_pattern` when auto-detection fails:
```bash
python diverse_performance.py --pred_pattern custom_pred_ --gt_pattern custom_gt_
```

## Architecture Overview

### Core Components
1. **Evaluation Scripts** (`diverse_performance.py`, `personalized_performance.py`)
   - Handle command-line arguments and orchestrate evaluation workflows
   - Support both automatic and manual configuration modes
   - Manage file discovery and validation across separated prediction/annotation directories

2. **Metrics Engine** (`metrics_set.py`)
   - `dice_at_all()`: Main function computing Dice variants with Hungarian algorithm optimal matching
   - `compute_multiclass_dice()`: Multi-class Dice with intelligent format detection and background exclusion
   - `generalized_energy_distance()`: Set-level similarity using energy distance formulation
   - Robust error handling for index mismatches and sparse class distributions

3. **Auto-Configuration System** (`auto_config.py`)
   - `analyze_dataset_automatically()`: Comprehensive dataset analysis and parameter inference
   - Intelligent file vs directory path handling with automatic pattern extraction
   - Sparse class detection and num_classes calculation using max_class + 1 methodology
   - Cross-validation between prediction and annotation compatibility

### Evaluation Methodologies
- **Diverse Performance**: Set-level metrics (GED, Dice_max, Dice_match, Dice_soft) measuring collective prediction-annotation similarity
- **Personalized Performance**: Individual expert Dice scores with statistical analysis (mean, std, min/max)
- **Hungarian Algorithm**: Optimal bipartite matching between predictions and annotations
- **Multi-rater Analysis**: Accounts for inter-annotator variability in medical segmentation tasks

### Multi-class Handling

#### Class Index Strategy
- **num_classes calculation**: Uses `max_class + 1` to handle sparse class indices
- **Example**: Classes [0, 2, 3] → num_classes = 4 (accommodates missing class 1)
- **Rationale**: Ensures proper one-hot encoding and framework compatibility

#### Background Processing
- **Default behavior**: Background class (0) excluded from evaluation metrics
- **Override option**: `--include_background` to include class 0 in calculations
- **Medical focus**: Aligns with clinical evaluation practices focusing on anatomical structures

### Data Pipeline
- **NIfTI I/O**: Uses `nibabel` for medical image format handling
- **Format Detection**: Automatic identification of class indices vs probability distributions
- **Validation**: Cross-checks prediction-annotation compatibility and spatial consistency
- **Error Recovery**: Robust handling of missing files, index mismatches, and format inconsistencies