# Multi-Rater Medical Image Segmentation Evaluation Toolkit

A specialized toolkit for evaluating medical image segmentation models against multiple annotator ground truth labels. This toolkit accounts for inter-annotator variability using advanced set-level similarity metrics.

## Quick Start

### 1. Install Dependencies
```bash
uv sync
```

### 2. Run Batch Evaluation
For evaluating all samples at once:
```bash
python competition_evaluation.py --pred_path predictions/ --gt_path MBH_val_label_2025/
```

### 3. Prepare Your Data

The toolkit automatically detects your data structure and parameters.

#### ID-based structure (like MBH dataset)

```
project/
├── predictions/
│   ├── ID_90ae3af3_ID_8d77fcb5d2.nii.gz
│   ├── ID_066b1fc2_ID_f937d7bff0.nii.gz
│   └── ID_0219ef88_ID_e5c1a31210.nii.gz
└── MBH_val_label_2025/
    ├── ID_90ae3af3_ID_8d77fcb5d2/
    │   ├── image.nii.gz
    │   ├── label_annot_1.nii.gz
    │   └── label_annot_5.nii.gz
    ├── ID_066b1fc2_ID_f937d7bff0/
    │   ├── image.nii.gz
    │   ├── label_annot_1.nii.gz
    │   └── label_annot_5.nii.gz
    └── ID_0219ef88_ID_e5c1a31210/
        ├── image.nii.gz
        ├── label_annot_1.nii.gz
        └── label_annot_5.nii.gz
```

**Note**: Prediction filenames must match the folder names in the labels directory for correct evaluation.

### 4. Basic Usage

For most evaluation needs, simply use:

```bash
python competition_evaluation.py --pred_path /path/to/predictions --gt_path /path/to/annotations
```

This command automatically:
- Detects all prediction files in your predictions directory
- Matches them with corresponding annotation folders
- Evaluates using appropriate metrics
- Handles ID-based naming (like MBH dataset structure)

## Evaluation Metrics

### Diverse Performance Metrics (Set-Level)
- **GED (Generalized Energy Distance)**: Set-level similarity between prediction and annotation sets
- **Dice_max**: Maximum Dice score achievable through optimal matching
- **Dice_match**: Dice score using Hungarian algorithm optimal matching
- **Dice_soft**: Soft Dice coefficient accounting for all annotation variations

### Personalized Performance Metrics
- **Individual Dice Scores**: Per-annotator Dice coefficients
- **Statistical Summary**: Mean, standard deviation, min/max across annotators

## Data Format Requirements

- **File Format**: NIfTI (`.nii.gz`)
- **Segmentation Maps**: Integer class indices (0, 1, 2, ...) or binary (0, 1)
- **Spatial Consistency**: All files must have matching dimensions
- **Class Handling**: Supports sparse class indices (e.g., [0, 2, 5])
- **ID Matching**: For MBH-style datasets, prediction filenames must exactly match the corresponding annotation folder names


## Output

Final evaluation results for all samples are saved in the `competition_results/` folder:
- `competition_aggregate_results.json` - Summary metrics across all samples
- `competition_detailed_results.json` - Detailed per-sample results
- `competition_report.txt` - Human-readable evaluation report
