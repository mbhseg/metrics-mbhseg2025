import nibabel as nib
import numpy as np
import os
from typing import Dict, Optional


def analyze_dataset_automatically(predictions_path: str, 
                                ground_truth_path: Optional[str] = None,
                                pred_pattern: str = 'pred_s', 
                                gt_pattern: str = 'label_annot_',
                                is_pred_file_specified: bool = False) -> Dict:
    """
    Automatically analyze dataset to extract parameters for evaluation.
    
    Args:
        predictions_path
        ground_truth_path
        pred_pattern
        gt_pattern
        
    Returns:
        A dictionary containing all automatically detected parameters
    """

    # Intelligent handling of file path vs directory path
    specified_pred_file = None
    if os.path.isfile(predictions_path):
        # If it is a file path, extract directory and filename
        pred_dir = os.path.dirname(predictions_path)
        # If dirname returns an empty string, use the current directory
        if not pred_dir:
            pred_dir = '.'
        specified_pred_file = os.path.basename(predictions_path)
        print(f"🔍 Detected prediction file: {specified_pred_file}")

        # User specified a specific prediction file, need to infer pattern from filename
        is_pred_file_specified = True

        # Infer pattern from prediction filename
        # Remove .nii.gz suffix, then remove numeric suffix to get pattern
        base_name = specified_pred_file.replace('.nii.gz', '')

        # Try several common pattern inference methods
        if pred_pattern not in base_name:
            # If the filename does not contain the default pattern, try to infer
            import re
            # First try removing the trailing numbers to get the pattern
            pattern_match = re.match(r'(.+?)(_\d+)?$', base_name)
            if pattern_match:
                inferred_pattern = pattern_match.group(1)
                # If the inferred pattern is reasonable (non-empty and contains characters), use it
                if inferred_pattern and len(inferred_pattern) > 2:
                    pred_pattern = inferred_pattern
                    print(f"Pattern inferred from filename: {pred_pattern}")
                else:
                    pred_pattern = base_name
                    print(f"Using full filename as prediction pattern: {pred_pattern}")

        predictions_path = pred_dir
    else:
        pred_dir = predictions_path

    # Process ground_truth_path
    if ground_truth_path is None:
        ground_truth_path = pred_dir
        # When a specific prediction file is specified, do not adjust gt_pattern, keep the default label_annot_
        # No need to print, as no annotation files have been found yet
    elif os.path.isfile(ground_truth_path):
        gt_dir = os.path.dirname(ground_truth_path)
        gt_filename = os.path.basename(ground_truth_path)
        print(f"🔍 Detected annotation file: {gt_filename}")

        # Only when the user explicitly specifies the annotation file, adjust gt_pattern
        if not gt_pattern in gt_filename:
            for possible_pattern in ['label_annot_', 'pred_annot_', 'label_', 'gt_', 'annotation_']:
                if possible_pattern in gt_filename:
                    gt_pattern = possible_pattern
                    print(f"🔄 Adjusted annotation pattern to: {gt_pattern}")
                    break
        
        ground_truth_path = gt_dir
    else:
        # ground_truth_path is a directory, keep gt_pattern unchanged
        pass

    # 1. Discover all files
    try:
        if is_pred_file_specified and specified_pred_file:
            # User specified a specific prediction file, only use that file
            pred_files = [specified_pred_file]
        else:
            # Use pattern to search for prediction files
            all_pred_files = os.listdir(predictions_path)
            pred_files = sorted([f for f in all_pred_files if pred_pattern in f and f.endswith('.nii.gz')])

        # Always use gt_pattern to search for annotation files
        if ground_truth_path == predictions_path:
            all_files = os.listdir(predictions_path)
            gt_files = sorted([f for f in all_files if gt_pattern in f and f.endswith('.nii.gz')])
        else:
            all_gt_files = os.listdir(ground_truth_path)
            gt_files = sorted([f for f in all_gt_files if gt_pattern in f and f.endswith('.nii.gz')])
    except Exception as e:
        raise ValueError(f"Cannot read directory {predictions_path}: {e}")

    print(f'🔍 Detected: {len(pred_files)} prediction files, {len(gt_files)} annotation files')

    # 2. Analyze annotation files to determine data characteristics
    all_classes = set()
    class_counts = {}
    total_pixels = 0
    file_shapes = []
    data_types = set()

    print('📊 Analyze annotation file characteristics...')
    for gt_file in gt_files:
        file_path = os.path.join(ground_truth_path, gt_file)
        if os.path.exists(file_path):
            data = nib.load(file_path).get_fdata()
            file_shapes.append(data.shape)
            data_types.add(str(data.dtype))
            
            unique_classes = np.unique(data)
            unique_classes = unique_classes[~np.isnan(unique_classes)]
            
            for cls in unique_classes:
                cls_int = int(cls)
                all_classes.add(cls_int)
                count = np.sum(data == cls)
                class_counts[cls_int] = class_counts.get(cls_int, 0) + count
                
            total_pixels += data.size

    # 3. Automatically infer parameters
    sorted_classes = sorted(list(all_classes))
    max_class = max(sorted_classes) if sorted_classes else 0

    # Smartly determine num_classes - always use max_class + 1 to ensure correct indexing
    num_classes = max_class + 1

    # Determine if binary: only [0,1] or [1] is considered binary
    is_binary = (sorted_classes == [0, 1] or sorted_classes == [1] or sorted_classes == [0])

    # Any other case is multiclass, even if there are only 2 classes but not contiguous (e.g. [0,5])
    if not is_binary:
        is_binary = False
    
    auto_params = {
        'all_classes': sorted_classes,
        'num_classes': num_classes,
        'max_class': max_class,
        'is_binary': is_binary,
        'is_multiclass': not is_binary,
        'has_background': 0 in all_classes,
        'class_distribution': class_counts,
        'total_pixels': total_pixels,
        'file_shapes': file_shapes,
        'data_types': list(data_types),
        'num_gt_files': len(gt_files),
        'num_pred_files': len(pred_files),
        'detected_pred_pattern': pred_pattern,
        'detected_gt_pattern': gt_pattern
    }

    # 4. Smartly suggest background handling
    if auto_params['has_background']:
        background_ratio = class_counts.get(0, 0) / total_pixels if total_pixels > 0 else 0
        auto_params['background_ratio'] = background_ratio
        # If background ratio exceeds 50%, suggest excluding background
        auto_params['exclude_background_recommended'] = background_ratio > 0.5
    else:
        auto_params['background_ratio'] = 0.0
        auto_params['exclude_background_recommended'] = False

    # 5. Validate predictions compatibility
    if pred_files:
        print('🔧 Check prediction files compatibility...')
        pred_classes = set()
        pred_shapes = []
        sample_count = min(3, len(pred_files))  # Check first 3 files
        
        for pred_file in pred_files[:sample_count]:
            file_path = os.path.join(predictions_path, pred_file)
            if os.path.exists(file_path):
                data = nib.load(file_path).get_fdata()
                pred_shapes.append(data.shape)
                unique_classes = np.unique(data)
                unique_classes = unique_classes[~np.isnan(unique_classes)]
                for cls in unique_classes:
                    pred_classes.add(int(cls))
                    
        auto_params['pred_classes'] = sorted(list(pred_classes))
        auto_params['pred_shapes'] = pred_shapes
        auto_params['pred_label_compatible'] = pred_classes.issubset(all_classes)
        auto_params['shapes_consistent'] = len(set(str(s) for s in file_shapes + pred_shapes)) == 1

        # Check if predictions might be in probability format
        if pred_files and len(pred_classes) > 0:
            max_pred_val = max(pred_classes) if pred_classes else 0
            min_pred_val = min(pred_classes) if pred_classes else 0
            auto_params['pred_seems_probabilities'] = (min_pred_val >= 0 and max_pred_val <= 1 and 
                                                     len(pred_classes) > 10)
        else:
            auto_params['pred_seems_probabilities'] = False
    
    return auto_params


def print_analysis_summary(params: Dict) -> None:
    """Print analysis summary"""
    print('\n' + '='*60)
    print('📋 Automatic Dataset Analysis Results')
    print('='*60)

    print(f"📁 Number of files: {params['num_pred_files']} predictions, {params['num_gt_files']} annotations")
    print(f"🏷️  Detected classes: {params['all_classes']}")
    print(f"📊  Number of actual classes: {len(params['all_classes'])}")
    
    if params['has_background']:
        bg_pct = params['background_ratio'] * 100
        print(f"🎯 Background class ratio: {bg_pct:.1f}%")

    # Explain the logic for calculating num_classes
    max_class = max(params['all_classes']) if params['all_classes'] else 0
    if len(params['all_classes']) != params['num_classes']:
        print(f"🔢 Recommended num_classes: {params['num_classes']} (max_class + 1 = {max_class} + 1, suitable for sparse classes)")
    else:
        print(f"🔢 Recommended num_classes: {params['num_classes']}")

    if params['is_binary']:
        print("📋 Recommended mode: Binary Segmentation")
    else:
        print("📋 Recommended mode: Multi-class Segmentation")

    if params.get('pred_label_compatible', True):
        print("✅ Prediction and annotation are compatible")
    else:
        print("⚠️  Prediction and annotation may be incompatible")
        print(f"   Prediction classes: {params.get('pred_classes', [])}")
        print(f"   Annotation classes: {params['all_classes']}")

    print('\n🚀 Recommended command line parameters:')
    if params['is_binary']:
        cmd = "# Binary mode (no additional parameters required)"
    else:
        cmd = f"--multiclass --num_classes {params['num_classes']}"
        
    if params.get('exclude_background_recommended', False):
        cmd += "  # Recommended to exclude background class"
    else:
        cmd += " --include_background"
    
    print(f"   {cmd}")


def get_auto_config(predictions_path: str,
                   ground_truth_path: Optional[str] = None,
                   pred_pattern: str = 'pred_s',
                   gt_pattern: str = 'label_annot_',
                   verbose: bool = True) -> Dict:
    """
    A convenient function to get automatic configuration.
    
    Returns:
        Automatically detected configuration parameters, ready for evaluation scripts.
    """
    # Detect if a specific prediction file is specified
    is_pred_file_specified = os.path.isfile(predictions_path)
    
    params = analyze_dataset_automatically(predictions_path, ground_truth_path, 
                                         pred_pattern, gt_pattern, is_pred_file_specified)
    
    if verbose:
        print_analysis_summary(params)
    
    # Return a simplified config dictionary
    config = {
        'multiclass': params['is_multiclass'],
        'num_classes': params['num_classes'],
        'exclude_background': params.get('exclude_background_recommended', True),
        'pred_pattern': params['detected_pred_pattern'],
        'gt_pattern': params['detected_gt_pattern']
    }
    
    return config, params


if __name__ == '__main__':
    # Test automatic configuration function
    config, details = get_auto_config('.', '.', 'pred_annot_', 'label_annot_')

    print(f"\n🎯 Final configuration: {config}")
