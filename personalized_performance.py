import torch
import numpy as np
import nibabel as nib
import os
import argparse
from tqdm import tqdm
from metrics_set import *
from auto_config import get_auto_config


def evaluate_personalized_performance(predictions_path, ground_truth_path, output_path=None,
                                     pred_pattern="pred_s", gt_pattern="label_annot_",
                                     num_predictions=None, num_annotators=None,
                                     multiclass=False, num_classes=2, exclude_background=True):
    """
    Calculate personalized performance metrics directly from saved predictions and ground truth.
    
    These metrics evaluate the Dice score for each individual expert:
    - Dice_each: Dice score for each expert/annotator
    - Dice_each_mean: Mean of all individual expert Dice scores
    
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
        Dictionary containing personalized performance metrics
    """
    
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
        pred_indices = set()
        for f in pred_files:
            if pred_pattern in f:
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
        annot_indices = set()
        for f in gt_files:
            if gt_pattern in f:
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
    print(f"Annotator IDs: {annotator_ids}")
    
    if num_predictions == 0 or num_annotators == 0:
        # Try to detect simple case structure (single case)
        if gt_files:  # If we found any ground truth files
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
    
    # Initialize metrics - flexible array size based on actual number of annotators
    Dice_each_accumulated = np.zeros(num_annotators)
    
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
            # Convert to tensors
            preds_tensor = torch.tensor(np.stack(preds)).unsqueeze(0).float()
            masks_tensor = torch.tensor(np.stack(masks)).unsqueeze(0).float()
            
            # Calculate personalized performance metrics
            _, _, _, dice_each_iter = dice_at_all(masks_tensor, preds_tensor, thresh=0.5,
                                                  multiclass=multiclass, num_classes=num_classes,
                                                  exclude_background=exclude_background)
            
            # Handle case where dice_each_iter might have different size
            if len(dice_each_iter) == num_annotators:
                Dice_each_accumulated = np.array(dice_each_iter)
            else:
                # Adjust if sizes don't match
                min_len = min(len(dice_each_iter), num_annotators)
                Dice_each_accumulated[:min_len] = dice_each_iter[:min_len]
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
                # Convert to tensors
                preds_tensor = torch.tensor(np.stack(preds)).unsqueeze(0).float()
                masks_tensor = torch.tensor(np.stack(masks)).unsqueeze(0).float()
                
                # Calculate personalized performance metrics
                _, _, _, dice_each_iter = dice_at_all(masks_tensor, preds_tensor, thresh=0.5,
                                                      multiclass=multiclass, num_classes=num_classes,
                                                      exclude_background=exclude_background)
                
                # Accumulate individual expert Dice scores
                min_len = min(len(dice_each_iter), num_annotators)
                Dice_each_accumulated[:min_len] += np.array(dice_each_iter[:min_len])
    
    # Calculate average metrics
    Dice_each_avg = Dice_each_accumulated / num_cases if num_cases > 0 else Dice_each_accumulated
    Dice_each_mean = np.mean(Dice_each_avg)
    
    # Build metrics dictionary
    metrics_dict = {
        'Dice_each': Dice_each_avg,
        'Dice_each_mean': Dice_each_mean,
    }
    
    # Add individual expert scores
    for i, annot_id in enumerate(annotator_ids):
        metrics_dict[f'Dice_expert_{annot_id}'] = Dice_each_avg[i]
    
    # Print results
    print("\n" + "="*60)
    print("PERSONALIZED PERFORMANCE METRICS")
    print("(Individual expert Dice scores)")
    if multiclass:
        print(f"Multi-class mode: {num_classes} classes")
        print(f"Background {'excluded' if exclude_background else 'included'}")
    print("="*60)
    print(f"{'Metric':<20s} {'Value':<10s}")
    print("-"*30)
    for i, annot_id in enumerate(annotator_ids):
        print(f"Dice_expert_{annot_id:<8}: {Dice_each_avg[i]:.4f}")
    print("-"*30)
    print(f"{'Dice_each_mean':<20s}: {Dice_each_mean:.4f}")
    print(f"{'Dice_each (array)':<20s}: [{', '.join([f'{v:.4f}' for v in Dice_each_avg])}]")
    
    # Save results if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("PERSONALIZED PERFORMANCE METRICS\n")
            f.write("(Individual expert Dice scores)\n")
            if multiclass:
                f.write(f"Multi-class mode: {num_classes} classes\n")
                f.write(f"Background {'excluded' if exclude_background else 'included'}\n")
            f.write("="*60 + "\n\n")
            
            # Write individual expert scores
            for i, annot_id in enumerate(annotator_ids):
                f.write(f"Dice_expert_{annot_id:<8}: {Dice_each_avg[i]:.4f}\n")
            
            f.write("-"*30 + "\n")
            f.write(f"Dice_each_mean      : {Dice_each_mean:.4f}\n")
            f.write(f"Dice_each (array)   : [{', '.join([f'{v:.4f}' for v in Dice_each_avg])}]\n")
            
            # Add statistics
            f.write("\n" + "="*60 + "\n")
            f.write("STATISTICS\n")
            f.write("-"*30 + "\n")
            f.write(f"Mean ± Std          : {Dice_each_mean:.4f} ± {np.std(Dice_each_avg):.4f}\n")
            f.write(f"Min / Max           : {np.min(Dice_each_avg):.4f} / {np.max(Dice_each_avg):.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write(f"Evaluated {num_cases} test cases\n")
            f.write(f"Number of annotators: {num_annotators}\n")
            f.write(f"Annotator IDs: {annotator_ids}\n")
            f.write(f"Predictions path: {predictions_path}\n")
            f.write(f"Ground truth path: {ground_truth_path}\n")
        print(f"\nResults saved to: {output_path}")
    
    return metrics_dict


def evaluate_personalized_performance_kfold(base_path, num_folds=5, output_path=None,
                                           multiclass=False, num_classes=2, exclude_background=True):
    """
    Evaluate personalized performance across multiple folds.
    
    Args:
        base_path: Base directory containing fold directories (e.g., results_0_fold/, results_1_fold/)
        num_folds: Number of folds to evaluate
        output_path: Optional path to save aggregated results
        multiclass: If True, use multi-class Dice calculation
        num_classes: Number of classes for multi-class segmentation
        exclude_background: If True, exclude background class from Dice calculation
    
    Returns:
        List of metric dictionaries for each fold
    """
    
    all_fold_results = []
    
    for fold_idx in range(num_folds):
        fold_path = os.path.join(base_path, f'results_{fold_idx}_fold/')
        
        if not os.path.exists(fold_path):
            print(f"Warning: Fold directory {fold_path} not found, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx + 1}/{num_folds}")
        print(f"{'='*60}")
        
        # Evaluate this fold
        metrics = evaluate_personalized_performance(fold_path, fold_path, None,
                                                   multiclass=multiclass,
                                                   num_classes=num_classes,
                                                   exclude_background=exclude_background)
        all_fold_results.append(metrics)
    
    if len(all_fold_results) > 0 and output_path:
        # Calculate mean and std across folds
        dice_each_all = np.array([result['Dice_each'] for result in all_fold_results])
        dice_mean_all = np.array([result['Dice_each_mean'] for result in all_fold_results])
        
        with open(output_path, 'w') as f:
            f.write("PERSONALIZED PERFORMANCE - K-FOLD RESULTS\n")
            if multiclass:
                f.write(f"Multi-class mode: {num_classes} classes\n")
                f.write(f"Background {'excluded' if exclude_background else 'included'}\n")
            f.write("="*60 + "\n\n")
            
            # Per-expert results across folds
            for expert_idx in range(4):
                expert_scores = dice_each_all[:, expert_idx]
                f.write(f"Dice_expert_{expert_idx+1}: {np.mean(expert_scores):.4f} ± {np.std(expert_scores):.4f}\n")
            
            f.write("-"*30 + "\n")
            f.write(f"Dice_each_mean: {np.mean(dice_mean_all):.4f} ± {np.std(dice_mean_all):.4f}\n")
            f.write(f"Dice_each: {np.mean(dice_each_all, axis=0)} ± {np.std(dice_each_all, axis=0)}\n")
            
        print(f"\nK-fold results saved to: {output_path}")
    
    return all_fold_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate personalized performance metrics for medical image segmentation')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to directory containing prediction files')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to directory containing ground truth files. If not provided, assumes same as pred_path')
    parser.add_argument('--output', type=str, default='personalized_performance_results.txt',
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
    parser.add_argument('--kfold', action='store_true',
                        help='Enable k-fold evaluation mode')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for k-fold evaluation')
    parser.add_argument('--auto_config', action='store_true',
                        help='Automatically detect dataset parameters (overrides manual settings)')
    
    args = parser.parse_args()
    
    # 自动配置检测
    if args.auto_config and not args.kfold:
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
    
    if args.kfold:
        # K-fold evaluation mode
        metrics = evaluate_personalized_performance_kfold(args.pred_path, args.num_folds, args.output,
                                                         multiclass=args.multiclass,
                                                         num_classes=args.num_classes,
                                                         exclude_background=not args.include_background)
    else:
        # Single evaluation mode
        gt_path = args.gt_path if args.gt_path else args.pred_path
        metrics = evaluate_personalized_performance(args.pred_path, gt_path, args.output,
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