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
    自动分析数据集，提取所有必要的参数
    
    Args:
        predictions_path: 预测文件路径或目录路径
        ground_truth_path: 标注文件路径或目录路径（可选，默认使用predictions_path）
        pred_pattern: 预测文件模式
        gt_pattern: 标注文件模式
        
    Returns:
        包含所有自动检测参数的字典
    """
    
    # 智能处理文件路径 vs 目录路径
    specified_pred_file = None
    if os.path.isfile(predictions_path):
        # 如果是文件路径，提取目录和文件名
        pred_dir = os.path.dirname(predictions_path)
        # 如果dirname返回空字符串，使用当前目录
        if not pred_dir:
            pred_dir = '.'
        specified_pred_file = os.path.basename(predictions_path)
        print(f"🔍 检测到预测文件: {specified_pred_file}")
        
        # 用户指定了具体预测文件，需要从文件名推断pattern
        is_pred_file_specified = True
        
        # 从预测文件名推断pattern
        # 去除.nii.gz后缀，然后去除数字后缀来得到pattern
        base_name = specified_pred_file.replace('.nii.gz', '')
        
        # 尝试几种常见的pattern推断方式
        if pred_pattern not in base_name:
            # 如果文件名不包含默认pattern，尝试推断
            import re
            # 先尝试去除结尾的数字得到pattern
            pattern_match = re.match(r'(.+?)(_\d+)?$', base_name)
            if pattern_match:
                inferred_pattern = pattern_match.group(1)
                # 如果推断的pattern合理（不为空且包含字符），使用它
                if inferred_pattern and len(inferred_pattern) > 2:
                    pred_pattern = inferred_pattern
                    print(f"🔄 从文件名推断预测pattern为: {pred_pattern}")
                else:
                    # 如果没有数字后缀，使用整个文件名作为pattern
                    pred_pattern = base_name
                    print(f"🔄 使用完整文件名作为预测pattern: {pred_pattern}")
        
        predictions_path = pred_dir
    else:
        pred_dir = predictions_path
    
    # 处理ground_truth_path
    if ground_truth_path is None:
        ground_truth_path = pred_dir
        # 当用户指定具体预测文件时，不调整gt_pattern，保持默认的label_annot_
        # 不需要打印，因为还没有发现标注文件
    elif os.path.isfile(ground_truth_path):
        gt_dir = os.path.dirname(ground_truth_path)
        gt_filename = os.path.basename(ground_truth_path)
        print(f"🔍 检测到标注文件: {gt_filename}")
        
        # 只有当用户明确指定标注文件时，才调整gt_pattern
        if not gt_pattern in gt_filename:
            for possible_pattern in ['label_annot_', 'pred_annot_', 'label_', 'gt_', 'annotation_']:
                if possible_pattern in gt_filename:
                    gt_pattern = possible_pattern
                    print(f"🔄 自动调整标注pattern为: {gt_pattern}")
                    break
        
        ground_truth_path = gt_dir
    else:
        # ground_truth_path是目录，保持gt_pattern不变
        pass
    
    # 1. 发现所有文件
    try:
        if is_pred_file_specified and specified_pred_file:
            # 用户指定了具体预测文件，只使用该文件
            pred_files = [specified_pred_file]
        else:
            # 用pattern搜索预测文件
            all_pred_files = os.listdir(predictions_path)
            pred_files = sorted([f for f in all_pred_files if pred_pattern in f and f.endswith('.nii.gz')])
        
        # 总是用gt_pattern搜索标注文件
        if ground_truth_path == predictions_path:
            all_files = os.listdir(predictions_path)
            gt_files = sorted([f for f in all_files if gt_pattern in f and f.endswith('.nii.gz')])
        else:
            all_gt_files = os.listdir(ground_truth_path)
            gt_files = sorted([f for f in all_gt_files if gt_pattern in f and f.endswith('.nii.gz')])
    except Exception as e:
        raise ValueError(f"无法读取目录 {predictions_path}: {e}")
    
    print(f'🔍 自动发现: {len(pred_files)} 个预测文件, {len(gt_files)} 个标注文件')
    
    # 2. 分析标注文件以确定数据特征
    all_classes = set()
    class_counts = {}
    total_pixels = 0
    file_shapes = []
    data_types = set()
    
    print('📊 分析标注文件特征...')
    for gt_file in gt_files:
        file_path = os.path.join(ground_truth_path, gt_file)
        if os.path.exists(file_path):
            data = nib.load(file_path).get_fdata()
            file_shapes.append(data.shape)
            data_types.add(str(data.dtype))
            
            # 统计类别
            unique_classes = np.unique(data)
            unique_classes = unique_classes[~np.isnan(unique_classes)]
            
            for cls in unique_classes:
                cls_int = int(cls)
                all_classes.add(cls_int)
                count = np.sum(data == cls)
                class_counts[cls_int] = class_counts.get(cls_int, 0) + count
                
            total_pixels += data.size
    
    # 3. 自动推断参数
    sorted_classes = sorted(list(all_classes))
    max_class = max(sorted_classes) if sorted_classes else 0
    
    # 智能确定num_classes
    if len(sorted_classes) <= 2:
        # 二进制情况
        num_classes = 2
        is_binary = True
    else:
        # 多类别情况：使用max_class + 1
        num_classes = max_class + 1
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
    
    # 4. 智能建议背景处理
    if auto_params['has_background']:
        background_ratio = class_counts.get(0, 0) / total_pixels if total_pixels > 0 else 0
        auto_params['background_ratio'] = background_ratio
        # 如果背景占比超过50%，建议排除背景
        auto_params['exclude_background_recommended'] = background_ratio > 0.5
    else:
        auto_params['background_ratio'] = 0.0
        auto_params['exclude_background_recommended'] = False
    
    # 5. 验证predictions兼容性
    if pred_files:
        print('🔧 检查预测文件兼容性...')
        pred_classes = set()
        pred_shapes = []
        sample_count = min(3, len(pred_files))  # 检查前3个文件
        
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
        
        # 检查预测是否可能是概率格式
        if pred_files and len(pred_classes) > 0:
            max_pred_val = max(pred_classes) if pred_classes else 0
            min_pred_val = min(pred_classes) if pred_classes else 0
            auto_params['pred_seems_probabilities'] = (min_pred_val >= 0 and max_pred_val <= 1 and 
                                                     len(pred_classes) > 10)
        else:
            auto_params['pred_seems_probabilities'] = False
    
    return auto_params


def print_analysis_summary(params: Dict) -> None:
    """打印分析结果摘要"""
    print('\n' + '='*60)
    print('📋 自动数据集分析结果')
    print('='*60)
    
    print(f"📁 文件数量: {params['num_pred_files']} 预测, {params['num_gt_files']} 标注")
    print(f"🏷️  检测到类别: {params['all_classes']}")
    print(f"📊 实际类别数量: {len(params['all_classes'])} 个类别")
    
    if params['has_background']:
        bg_pct = params['background_ratio'] * 100
        print(f"🎯 背景类占比: {bg_pct:.1f}%")
    
    # 解释num_classes的计算逻辑
    max_class = max(params['all_classes']) if params['all_classes'] else 0
    if len(params['all_classes']) != params['num_classes']:
        print(f"🔢 推荐 num_classes: {params['num_classes']} (max_class + 1 = {max_class} + 1, 适用于稀疏类别)")
    else:
        print(f"🔢 推荐 num_classes: {params['num_classes']}")
    
    if params['is_binary']:
        print("📋 推荐模式: 二进制分割")
    else:
        print("📋 推荐模式: 多类别分割")
    
    if params.get('pred_label_compatible', True):
        print("✅ 预测与标注兼容")
    else:
        print("⚠️  预测与标注可能不兼容")
        print(f"   预测类别: {params.get('pred_classes', [])}")
        print(f"   标注类别: {params['all_classes']}")
    
    print('\n🚀 建议的命令行参数:')
    if params['is_binary']:
        cmd = "# 二进制模式 (无需额外参数)"
    else:
        cmd = f"--multiclass --num_classes {params['num_classes']}"
        
    if params.get('exclude_background_recommended', False):
        cmd += "  # 建议排除背景类"
    else:
        cmd += " --include_background"
    
    print(f"   {cmd}")


def get_auto_config(predictions_path: str,
                   ground_truth_path: Optional[str] = None,
                   pred_pattern: str = 'pred_s',
                   gt_pattern: str = 'label_annot_',
                   verbose: bool = True) -> Dict:
    """
    获取自动配置的便捷函数
    
    Returns:
        自动检测的配置参数，可直接用于评估脚本
    """
    # 检测是否指定了具体文件
    is_pred_file_specified = os.path.isfile(predictions_path)
    
    params = analyze_dataset_automatically(predictions_path, ground_truth_path, 
                                         pred_pattern, gt_pattern, is_pred_file_specified)
    
    if verbose:
        print_analysis_summary(params)
    
    # 返回可直接使用的配置
    config = {
        'multiclass': params['is_multiclass'],
        'num_classes': params['num_classes'],
        'exclude_background': params.get('exclude_background_recommended', True),
        'pred_pattern': params['detected_pred_pattern'],
        'gt_pattern': params['detected_gt_pattern']
    }
    
    return config, params


if __name__ == '__main__':
    # 测试自动配置功能
    config, details = get_auto_config('.', '.', 'pred_annot_', 'label_annot_')
    
    print(f"\n🎯 最终配置: {config}")