import torch
import numpy as np
import nibabel as nib
import os
import argparse
from tqdm import tqdm
from metrics_set import *
from auto_config import get_auto_config

from metrics_set import DEVICE


def evaluate_diverse_performance(predictions_path, ground_truth_path, output_path=None, 
                                pred_pattern="pred_s", gt_pattern="label_annot_",
                                num_predictions=None, num_annotators=None,
                                multiclass=False, num_classes=2, exclude_background=True):
    """
    Calculate diverse performance metrics directly from saved predictions and ground truth.
    
    These metrics evaluate set-level similarity between the prediction set and label set:
    - GED score: Generalized Energy Distance
    - Dice_soft score: Soft Dice coefficient
    - Dice_match: Set-level matching metric
    - Dice_max and Dice_max_reverse: Set-level maximum metrics
    
    Args:
        predictions_path: Directory containing prediction files
        ground_truth_path: Directory containing ground truth files
        output_path: Optional path to save results
        pred_pattern: Pattern for prediction files (default: "pred_s")
        gt_pattern: Pattern for ground truth files (default: "label_annot_")
        num_predictions: Number of predictions per case (auto-detect if None)
        num_annotators: Number of annotators (auto-detect if None)
        multiclass: If True, use multi-class Dice calculation
        num_classes: Number of classes for multi-class segmentation
        exclude_background: If True, exclude background class from Dice calculation
    
    Returns:
        Dictionary containing all diverse performance metrics
    """
    
    # Check if predictions_path is a file instead of directory
    if os.path.isfile(predictions_path):
        raise ValueError(
            f"❌ 检测到文件路径而非目录路径: {predictions_path}\n"
            f"💡 解决方案: 请使用 --auto_config 参数进行自动配置\n"
            f"   示例: python diverse_performance.py --pred_path {predictions_path} --gt_path {ground_truth_path} --auto_config"
        )
    
    # Auto-detect file patterns and counts
    pred_all_files = os.listdir(predictions_path)
    
    # Find prediction files in prediction directory
    pred_files = sorted([f for f in pred_all_files if pred_pattern in f and f.endswith('.nii.gz')])
    
    # Find annotator files in ground truth directory (might be different from predictions)
    if ground_truth_path == predictions_path:
        gt_files = sorted([f for f in pred_all_files if gt_pattern in f and f.endswith('.nii.gz')])
    else:
        gt_all_files = os.listdir(ground_truth_path)
        gt_files = sorted([f for f in gt_all_files if gt_pattern in f and f.endswith('.nii.gz')])
    
    # Auto-detect number of predictions and annotators if not specified
    if num_predictions is None:
        # Count unique prediction indices
        pred_indices = set()
        for f in pred_files:
            if pred_pattern in f:
                # Extract number after pattern
                try:
                    idx = f.split(pred_pattern)[-1].split('.')[0].split('_')[0]
                    pred_indices.add(idx)
                except:
                    pass
        num_predictions = len(pred_indices) if pred_indices else 0
        pred_ids = sorted(list(pred_indices))
    else:
        pred_ids = list(range(1, num_predictions + 1))
    
    if num_annotators is None:
        # Count unique annotator indices
        annot_indices = set()
        for f in gt_files:
            if gt_pattern in f:
                # Extract number after pattern
                try:
                    idx = f.split(gt_pattern)[-1].split('.')[0].split('_')[0]
                    annot_indices.add(idx)
                except:
                    pass
        num_annotators = len(annot_indices) if annot_indices else 0
        annotator_ids = sorted(list(annot_indices))
    else:
        annotator_ids = list(range(1, num_annotators + 1))
    
    print(f"Detected {num_predictions} predictions and {num_annotators} annotators")
    
    if num_predictions == 0 or num_annotators == 0:
        # Try to detect simple case structure (single case)
        if gt_files:  # If we found any ground truth files
            # Single case mode
            num_cases = 1
            print("Single case mode detected")
        else:
            raise ValueError(f"Could not detect valid files. Predictions in {predictions_path}, Ground truth in {ground_truth_path}")
    else:
        # Multi-case mode - detect number of cases
        case_ids = set()
        for f in pred_files:
            case_id = f.split('_')[0] if '_' in f else "single"
            case_ids.add(case_id)
        num_cases = len(case_ids)
    
    # Initialize metrics
    GED_global = 0.0
    Dice_max = 0.0
    Dice_max_reverse = 0.0
    Dice_soft = 0.0
    Dice_match = 0.0
    
    print(f"Evaluating {num_cases} test case(s) with {num_annotators} annotator(s)...")
    
    if num_cases == 1:
        # Single case - load all predictions and annotations
        preds = []
        masks = []
        
        # Load predictions
        for pred_id in pred_ids:
            pred_file = os.path.join(predictions_path, f"{pred_pattern}{pred_id}.nii.gz")
            if os.path.exists(pred_file):
                pred_img = nib.load(pred_file).get_fdata()
                preds.append(pred_img)
        
        # Load ground truth masks
        for annot_id in annotator_ids:
            mask_file = os.path.join(ground_truth_path, f"{gt_pattern}{annot_id}.nii.gz")
            if os.path.exists(mask_file):
                mask_img = nib.load(mask_file).get_fdata()
                masks.append(mask_img)
        
        if len(preds) > 0 and len(masks) > 0:
            # Convert to tensors and move to GPU
            preds_tensor = torch.tensor(np.stack(preds)).unsqueeze(0).float().to(DEVICE)
            masks_tensor = torch.tensor(np.stack(masks)).unsqueeze(0).float().to(DEVICE)
            
            print(f"📊 使用 {DEVICE} 计算指标...")
            
            # Calculate metrics - 所有计算都在GPU上进行
            GED_global = generalized_energy_distance(masks_tensor, preds_tensor, num_classes=num_classes)
            dice_max, dice_max_reverse, dice_match, _ = dice_at_all(masks_tensor, preds_tensor, thresh=0.5,
                                                                     multiclass=multiclass, num_classes=num_classes,
                                                                     exclude_background=exclude_background)
            dice_soft = dice_at_thresh(masks_tensor, preds_tensor)
            
            Dice_max = dice_max
            Dice_max_reverse = dice_max_reverse
            Dice_match = dice_match
            Dice_soft = dice_soft
    else:
        # Multi-case mode
        for case_id in tqdm(sorted(case_ids)):
            preds = []
            masks = []
            
            # Load predictions for this case
            for pred_id in pred_ids:
                pred_file = os.path.join(predictions_path, f"{case_id}_{pred_pattern}{pred_id}.nii.gz")
                if os.path.exists(pred_file):
                    pred_img = nib.load(pred_file).get_fdata()
                    preds.append(pred_img)
            
            # Load ground truth masks for this case
            for annot_id in annotator_ids:
                mask_file = os.path.join(ground_truth_path, f"{case_id}_{gt_pattern}{annot_id}.nii.gz")
                if not os.path.exists(mask_file):
                    mask_file = os.path.join(predictions_path, f"{case_id}_{gt_pattern}{annot_id}.nii.gz")
                
                if os.path.exists(mask_file):
                    mask_img = nib.load(mask_file).get_fdata()
                    masks.append(mask_img)
            
            if len(preds) > 0 and len(masks) > 0:
                # Convert to tensors and move to GPU
                preds_tensor = torch.tensor(np.stack(preds)).unsqueeze(0).float().to(DEVICE)
                masks_tensor = torch.tensor(np.stack(masks)).unsqueeze(0).float().to(DEVICE)
                
                # Calculate diverse performance metrics - GPU加速
                GED_iter = generalized_energy_distance(masks_tensor, preds_tensor, num_classes=num_classes)
                dice_max_iter, dice_max_reverse_iter, dice_match_iter, _ = dice_at_all(masks_tensor, preds_tensor, thresh=0.5,
                                                                                       multiclass=multiclass, num_classes=num_classes,
                                                                                       exclude_background=exclude_background)
                dice_soft_iter = dice_at_thresh(masks_tensor, preds_tensor)
                
                GED_global += GED_iter
                Dice_match += dice_match_iter
                Dice_max += dice_max_iter
                Dice_max_reverse += dice_max_reverse_iter
                Dice_soft += dice_soft_iter
    
    # Calculate average metrics
    metrics_dict = {
        'GED': GED_global / num_cases,
        'Dice_max': Dice_max / num_cases,
        'Dice_max_reverse': Dice_max_reverse / num_cases,
        'Dice_max_mean': (Dice_max_reverse + Dice_max) / (2 * num_cases),
        'Dice_match': Dice_match / num_cases,
        'Dice_soft': Dice_soft / num_cases
    }
    
    # Print results
    print("\n" + "="*60)
    print("DIVERSE PERFORMANCE METRICS")
    print("(Set-level similarity between predictions and labels)")
    if multiclass:
        print(f"Multi-class mode: {num_classes} classes")
        print(f"Background {'excluded' if exclude_background else 'included'}")
        print("Note: Dice_soft currently uses binary calculation")
    print("="*60)
    for key, value in metrics_dict.items():
        print(f"{key:20s}: {value:.4f}")
    
    # Save results if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("DIVERSE PERFORMANCE METRICS\n")
            f.write("(Set-level similarity between predictions and labels)\n")
            if multiclass:
                f.write(f"Multi-class mode: {num_classes} classes\n")
                f.write(f"Background {'excluded' if exclude_background else 'included'}\n")
                f.write("Note: Dice_soft currently uses binary calculation\n")
            f.write("="*60 + "\n\n")
            for key, value in metrics_dict.items():
                f.write(f"{key:20s}: {value:.4f}\n")
            f.write("\n" + "="*60 + "\n")
            f.write(f"Evaluated {num_cases} test cases\n")
            f.write(f"Predictions path: {predictions_path}\n")
            f.write(f"Ground truth path: {ground_truth_path}\n")
        print(f"\nResults saved to: {output_path}")
    
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description='Evaluate diverse performance metrics for medical image segmentation')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to directory containing prediction files')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to directory containing ground truth files. If not provided, assumes same as pred_path')
    parser.add_argument('--output', type=str, default='diverse_performance_results.txt',
                        help='Output file path for results')
    parser.add_argument('--pred_pattern', type=str, default='pred_s',
                        help='Pattern for prediction files (e.g., "pred_s" for pred_s1.nii.gz)')
    parser.add_argument('--gt_pattern', type=str, default='label_annot_',
                        help='Pattern for ground truth files (e.g., "label_annot_" for label_annot_1.nii.gz)')
    parser.add_argument('--num_pred', type=int, default=None,
                        help='Number of predictions per case (auto-detect if not specified)')
    parser.add_argument('--num_annot', type=int, default=None,
                        help='Number of annotators (auto-detect if not specified)')
    parser.add_argument('--multiclass', action='store_true',
                        help='Enable multi-class segmentation mode')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes for multi-class segmentation (default: 2 for binary)')
    parser.add_argument('--include_background', action='store_true',
                        help='Include background class (class 0) in Dice calculation')
    parser.add_argument('--auto_config', action='store_true',
                        help='Automatically detect dataset parameters (overrides manual settings)')
    
    args = parser.parse_args()
    
    # 自动配置检测
    if args.auto_config:
        print("🤖 启用自动配置检测...")
        auto_config, auto_details = get_auto_config(
            args.pred_path, 
            args.gt_path if args.gt_path else args.pred_path,
            args.pred_pattern, 
            args.gt_pattern,
            verbose=True
        )
        
        # 使用自动检测的参数
        args.multiclass = auto_config['multiclass']
        args.num_classes = auto_config['num_classes']
        args.include_background = not auto_config['exclude_background']
        args.pred_pattern = auto_config['pred_pattern']
        args.gt_pattern = auto_config['gt_pattern']
        
        # 重要：如果输入是文件路径，使用目录路径进行评估
        if os.path.isfile(args.pred_path):
            pred_dir = os.path.dirname(args.pred_path)
            if not pred_dir:
                pred_dir = '.'
            args.pred_path = pred_dir
            print(f"🔄 输入文件路径已转换为目录路径: {args.pred_path}")
        
        if args.gt_path and os.path.isfile(args.gt_path):
            gt_dir = os.path.dirname(args.gt_path)
            if not gt_dir:
                gt_dir = '.'
            args.gt_path = gt_dir
            print(f"🔄 标注文件路径已转换为目录路径: {args.gt_path}")
        
        print(f"\n✅ 自动配置已应用:")
        print(f"   multiclass: {args.multiclass}")
        print(f"   num_classes: {args.num_classes}")
        print(f"   exclude_background: {not args.include_background}")
        print(f"   pred_pattern: {args.pred_pattern}")
        print(f"   gt_pattern: {args.gt_pattern}")
        print()
    
    # If ground truth path not provided, use same as predictions path
    gt_path = args.gt_path if args.gt_path else args.pred_path
    
    # Run evaluation
    metrics = evaluate_diverse_performance(args.pred_path, gt_path, args.output,
                                         pred_pattern=args.pred_pattern,
                                         gt_pattern=args.gt_pattern,
                                         num_predictions=args.num_pred,
                                         num_annotators=args.num_annot,
                                         multiclass=args.multiclass,
                                         num_classes=args.num_classes,
                                         exclude_background=not args.include_background)
    
    return metrics


if __name__ == '__main__':
    main()