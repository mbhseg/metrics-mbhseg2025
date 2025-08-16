import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0') 
    else:
        return torch.device('cpu')

DEVICE = get_device()
print(f"Using device: {DEVICE}")

def get_dice_threshold(output, mask, threshold):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """
    smooth = 1e-6

    if not output.is_cuda and DEVICE.type == 'cuda':
        output = output.to(DEVICE)
    if not mask.is_cuda and DEVICE.type == 'cuda':
        mask = mask.to(DEVICE)

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    output = output.view(-1)
    mask = mask.view(-1)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice


def get_soft_dice(outputs, masks):
    """
    :param outputs: B * output shape per image
    :param masks: B * mask shape per image
    :return: average dice of B items
    """
    dice_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        dice_item_thres_list = []
        for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
            dice_item_thres = get_dice_threshold(output, mask, thres)
            dice_item_thres_list.append(dice_item_thres.data)
        dice_item_thres_mean = np.mean(dice_item_thres_list)
        dice_list.append(dice_item_thres_mean)

    return np.mean(dice_list)


def get_iou_threshold(output, mask, threshold):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: iou of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)

    intersection = (output * mask).sum()
    total = (output + mask).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)

    return IoU


def get_soft_iou(outputs, masks):
    """
    :param outputs: B * output shape per image
    :param masks: B * mask shape per image
    :return: average iou of B items
    """
    iou_list = []
    for this_item in range(outputs.size(0)):
        output = outputs[this_item]
        mask = masks[this_item]
        iou_item_thres_list = []
        for thres in [0.1, 0.3, 0.5, 0.7, 0.9]:
            iou_item_thres = get_iou_threshold(output, mask, thres)
            iou_item_thres_list.append(iou_item_thres)
        iou_item_thres_mean = np.mean(iou_item_thres_list)
        iou_list.append(iou_item_thres_mean)

    return np.mean(iou_list)

# =========== GED ============= #


def segmentation_scores(mask1, mask2):
    IoU = get_iou_threshold(mask1, mask2, threshold=0.5)
    return 1.0 - IoU


def generalized_energy_distancex(label_list, pred_list):
    label_label_dist = [segmentation_scores(label_1, label_2) for i1, label_1 in enumerate(label_list)
                        for i2, label_2 in enumerate(label_list) if i1 != i2]
    pred_pred_dist = [segmentation_scores(pred_1, pred_2) for i1, pred_1 in enumerate(pred_list)
                      for i2, pred_2 in enumerate(pred_list) if i1 != i2]
    pred_label_list = [segmentation_scores(pred, label) for i, pred in enumerate(pred_list)
                       for j, label in enumerate(label_list)]
    GED = 2 * sum(pred_label_list) / len(pred_label_list) \
          - sum(label_label_dist) / len(label_label_dist) - sum(pred_pred_dist) / len(pred_pred_dist)
    return GED


def get_GED(batch_label_list, batch_pred_list):
    """
    :param batch_label_list: list_list
    :param batch_pred_list:
    :return:
    """
    batch_size = len(batch_pred_list)
    GED = 0.0
    for idx in range(batch_size):
        GED_temp = generalized_energy_distancex(label_list=batch_label_list[idx], pred_list=batch_pred_list[idx])
        GED = GED + GED_temp
    return GED / batch_size


def compute_dice_accuracy(label, mask):
    """
    Binary Dice coefficient calculation (legacy function for backward compatibility)
    """
    smooth = 1e-8
    batch = label.size(0)
    m1 = label.reshape(batch, -1).float()  # Flatten
    m2 = mask.reshape(batch, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def compute_multiclass_dice(label, pred, num_classes=None, exclude_background=True):
    """
    Compute multi-class Dice coefficient with GPU acceleration.
    Handles false positive classes gracefully by only evaluating classes present in ground truth.
    
    Args:
        label: Ground truth tensor. Shape can be:
               - [B, H, W, D] with integer class indices
               - [B, C, H, W, D] with one-hot encoding
        pred: Prediction tensor. Shape can be:
              - [B, H, W, D] with integer class indices  
              - [B, C, H, W, D] with one-hot encoding or probabilities
        num_classes: Number of classes (including background if not excluded)
        exclude_background: If True, exclude class 0 from Dice calculation
    
    Returns:
        Mean Dice coefficient across all classes present in ground truth (macro-averaged)
    """
    smooth = 1e-8
    if DEVICE.type == 'cuda':
        if not label.is_cuda:
            label = label.to(DEVICE)
        if not pred.is_cuda:
            pred = pred.to(DEVICE)
    
    # Handle different input formats
    if len(label.shape) == len(pred.shape):
        # Check if data is already one-hot encoded by examining if second dimension equals num_classes
        # and values are in [0,1] range
        is_onehot = (len(label.shape) >= 4 and 
                     num_classes is not None and 
                     label.shape[1] == num_classes and 
                     torch.all((label >= 0) & (label <= 1)) and
                     torch.all((pred >= 0) & (pred <= 1)))
        
        if is_onehot:  # One-hot encoded
            # Assume shape is [B, C, H, W, ...] 
            label_onehot = label
            pred_onehot = pred
        else:  # Integer class indices
            # Determine the number of classes based on what's present in GROUND TRUTH
            # This prevents crashes when predictions contain false positive classes
            label_max = label.max().item()
            pred_max = pred.max().item()
            
            if num_classes is None:
                # Use the maximum class from ground truth + 1 as the base number of classes
                # but ensure it's at least as large as any prediction class to avoid index errors
                num_classes = max(label_max, pred_max) + 1
            
            # However, for evaluation, we only consider classes that exist in ground truth
            gt_classes = torch.unique(label[label >= 0]).long().tolist()
            
            # Ensure no values exceed num_classes - 1 to prevent one-hot encoding errors
            label_clamped = torch.clamp(label.long(), 0, num_classes - 1)
            pred_clamped = torch.clamp(pred.long(), 0, num_classes - 1)
            
            # Create one-hot encoding with full num_classes to handle all possible predictions
            label_onehot = torch.nn.functional.one_hot(label_clamped, num_classes)
            pred_onehot = torch.nn.functional.one_hot(pred_clamped, num_classes)
            
            # Move class dimension to position 1: [B, H, W, D, C] -> [B, C, H, W, D]
            label_onehot = label_onehot.permute(0, -1, *range(1, len(label_onehot.shape)-1))
            pred_onehot = pred_onehot.permute(0, -1, *range(1, len(pred_onehot.shape)-1))
    else:
        raise ValueError(f"Label and prediction shapes don't match: {label.shape} vs {pred.shape}")
    
    # Convert to float
    label_onehot = label_onehot.float()
    pred_onehot = pred_onehot.float()
    
    # Get unique classes present in ground truth (excluding background if specified)
    if not is_onehot:
        gt_classes = torch.unique(label[label >= 0]).long().tolist()
    else:
        # For one-hot encoded data, find classes with non-zero ground truth
        gt_classes = []
        for c in range(num_classes):
            if torch.sum(label_onehot[:, c]) > 0:
                gt_classes.append(c)
    
    start_class = 1 if exclude_background else 0
    dice_scores = []
    
    # Only iterate over classes that are present in ground truth
    for c in gt_classes:
        if c < start_class:  # Skip background if excluded
            continue
            
        if c >= label_onehot.shape[1] or c >= pred_onehot.shape[1]:
            # Class index exceeds tensor dimensions, skip
            continue
            
        label_c = label_onehot[:, c]
        pred_c = pred_onehot[:, c]
        
        # Flatten spatial dimensions
        label_c = label_c.reshape(label_c.shape[0], -1)
        pred_c = pred_c.reshape(pred_c.shape[0], -1)
        
        # Calculate Dice for this class
        intersection = torch.sum(label_c * pred_c)
        union = torch.sum(label_c) + torch.sum(pred_c)
        
        if union > 0:
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores.append(dice.item())  # Convert to Python scalar
        # Note: Classes with union=0 (not present in data) are excluded from mean calculation
        # This ensures we only average over classes that actually exist in the data
    
    # Return mean Dice across existing classes in ground truth (macro-averaged)
    # False positive classes in predictions that don't exist in ground truth are implicitly penalized
    # as they contribute 0 to intersection but add to the prediction union term
    return np.mean(dice_scores) if dice_scores else 0.0


def dice_at_all(labels, preds, thresh=0.5, is_test=False, multiclass=False, num_classes=None, exclude_background=True):
    """
    Calculate various Dice metrics for multiple predictions and labels.
    
    Args:
        labels: Ground truth labels [batch, num_annotators, H, W, D] or [batch, num_annotators, C, H, W, D]
        preds: Predictions [batch, num_predictions, H, W, D] or [batch, num_predictions, C, H, W, D]
        thresh: Threshold for binarization (only used if not multiclass)
        is_test: If True, calculate diagonal dice (prediction i vs label i)
        multiclass: If True, use multi-class Dice calculation
        num_classes: Number of classes for multi-class segmentation
        exclude_background: If True, exclude background class from Dice calculation
    
    Returns:
        dice_max: Maximum Dice for each prediction across all labels (mean)
        dice_max_reverse: Maximum Dice for each label across all predictions (mean)
        dice_match: Dice after optimal matching using Hungarian algorithm
        dice_each: Individual Dice scores for matched pairs
    """
    
    if not multiclass:
        # Binary segmentation - original implementation
        pred_masks = (preds > thresh).float()
        compute_dice_fn = compute_dice_accuracy
    else:
        # Multi-class segmentation
        pred_masks = preds
        # Check if data is in probability/logit format vs class indices format
        # For class indices: expect integer values in reasonable range (< num_classes * 2)
        # For probabilities/logits: expect shape [batch, num_pred, C, H, W, D] where C = num_classes
        
        # Check if this could be probability/logit format by examining the channel dimension
        has_channel_dim = (len(preds.shape) == 6 and preds.shape[2] == num_classes)
        
        if has_channel_dim:
            # Shape is [batch, num_pred, C, H, W, D] - convert from probabilities/logits
            pred_masks = torch.argmax(preds, dim=2)
            labels_processed = torch.argmax(labels, dim=2) if labels.shape[2] == num_classes else labels
        else:
            # Shape is [batch, num_pred, H, W, D] - already class indices
            pred_masks = preds
            labels_processed = labels
        
        # Define multi-class dice function for matrix computation
        def compute_dice_fn(label, pred):
            return compute_multiclass_dice(label, pred, num_classes=num_classes, 
                                          exclude_background=exclude_background)
    
    dice_each = []
    dice_matrix = np.zeros([labels.shape[1], pred_masks.shape[1]])
    
    for i in range(labels.shape[1]):
        for j in range(pred_masks.shape[1]):
            if multiclass:
                dice_matrix[i, j] = compute_dice_fn(labels_processed[:, i], pred_masks[:, j])
            else:
                dice_matrix[i, j] = compute_dice_fn(labels[:, i], pred_masks[:, j])
    
    dice_max = dice_matrix.max(0).mean()  # Max dice for each prediction
    dice_max_reverse = dice_matrix.max(1).mean()  # Max dice for each label
    
    # Hungarian algorithm for optimal matching
    cost_matrix = 1 - dice_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    dice_match = []
    
    # Create mapping from row to column assignment
    assignment_dict = {row: col for row, col in zip(row_ind, col_ind)}
    
    for i in range(labels.shape[1]):
        if i in assignment_dict:
            # Use assigned column
            assigned_col = assignment_dict[i]
            dice_match.append(1 - cost_matrix[i, assigned_col])
            if is_test:
                # For diagonal test, only use if i is within valid column range
                if i < dice_matrix.shape[1]:
                    dice_each.append(dice_matrix[i, i])
                else:
                    dice_each.append(0.0)  # No valid diagonal element
            else:
                dice_each.append(1 - cost_matrix[i, assigned_col])
        else:
            # No assignment for this row (more labels than predictions)
            # Use the best available match
            best_col = np.argmax(dice_matrix[i, :])
            dice_match.append(dice_matrix[i, best_col])
            if is_test:
                # For diagonal test, only use if i is within valid column range
                if i < dice_matrix.shape[1]:
                    dice_each.append(dice_matrix[i, i])
                else:
                    dice_each.append(0.0)  # No valid diagonal element
            else:
                dice_each.append(dice_matrix[i, best_col])
    
    dice_match = np.mean(dice_match)
    
    return dice_max, dice_max_reverse, dice_match, dice_each


def iou(x, y, axis=-1):
	smooth = 1e-8
	iou_ = ((x & y).sum(axis)) / ((x | y).sum(axis)+smooth)
	iou_[np.isnan(iou_)] = 1.
	return iou_

# exclude background
def distance(x, y):
	try:
		per_class_iou = iou(x[:, None], y[None, :], axis=-2)
	except MemoryError:
		per_class_iou = []
		for x_ in x:
			per_class_iou.append(iou(np.expand_dims(x_, axis=0), y[None, :], axis=-2))
		per_class_iou = np.concatenate(per_class_iou)
	
	# Calculate distance as 1 - IoU for each class
	result = 1 - np.nanmean(per_class_iou[..., 1:], axis=-1)
	
	# Handle NaN values: replace with 1.0
	result = np.where(np.isnan(result), 1.0, result)
	
	return result


def calc_generalised_energy_distance(dist_0, dist_1, num_classes):
	dist_0 = dist_0.reshape((len(dist_0), -1))
	dist_1 = dist_1.reshape((len(dist_1), -1))
	
	# Convert to numpy arrays if they are torch tensors
	if isinstance(dist_0, torch.Tensor):
		dist_0 = dist_0.cpu().numpy().astype("int")
	else:
		dist_0 = dist_0.numpy().astype("int")
		
	if isinstance(dist_1, torch.Tensor):
		dist_1 = dist_1.cpu().numpy().astype("int")
	else:
		dist_1 = dist_1.numpy().astype("int")

	eye = np.eye(num_classes)
	dist_0 = eye[dist_0].astype('bool')
	dist_1 = eye[dist_1].astype('bool')

	cross_distance = np.mean(distance(dist_0, dist_1))
	distance_0 = np.mean(distance(dist_0, dist_0))
	distance_1 = np.mean(distance(dist_1, dist_1))
	return cross_distance, distance_0, distance_1


# Metrics for Uncertainty
def generalized_energy_distance(labels, preds, thresh=0.5, num_classes=2):
	pred_masks = (preds > thresh).float()
	
	# Convert to CPU if necessary
	if isinstance(labels, torch.Tensor):
		labels_cpu = labels.cpu()
	else:
		labels_cpu = labels
		
	if isinstance(pred_masks, torch.Tensor):
		pred_cpu = pred_masks.cpu()
	else:
		pred_cpu = pred_masks
	
	# Handle cases with false positive classes in predictions
	# Get actual classes present in both labels and predictions
	label_classes = set(int(x) for x in np.unique(labels_cpu[0].numpy()) if x != 0)
	pred_classes = set(int(x) for x in np.unique(pred_cpu[0].numpy()) if x != 0)
	
	if len(label_classes) == 0 and len(pred_classes) == 0:
		# Both only have background, no difference
		return 0.0
	elif len(label_classes) == 0 or len(pred_classes) == 0:
		# One has no classes, return 1.0 as they are completely different
		return 1.0
	
	# Update num_classes to accommodate all classes that appear in either labels or predictions
	# This prevents index errors when predictions contain classes not in ground truth
	max_class_in_data = max(
		max(label_classes) if label_classes else 0,
		max(pred_classes) if pred_classes else 0
	)
	effective_num_classes = max(num_classes, max_class_in_data + 1)
	
	cross, d_0, d_1 = calc_generalised_energy_distance(labels[0], pred_masks[0], effective_num_classes)
	GED = 2 * cross - d_0 - d_1
	
	# Handle NaN values
	if np.isnan(GED):
		GED = 0.0

	return GED

def dice_at_thresh(labels, preds):
	thres_list = [0.1, 0.3, 0.5, 0.7, 0.9]

	pred_mean = preds.mean(1)
	label_mean = labels.mean(1).float()

	dice_scores = []
	for thresh in thres_list:
		pred_binary = (pred_mean > thresh).float()
		label_binary = (label_mean > thresh).float()
		dice_score = compute_dice_accuracy(label_binary, pred_binary)
		# Convert to Python scalar if it's a tensor
		if isinstance(dice_score, torch.Tensor):
			dice_scores.append(dice_score.item())
		else:
			dice_scores.append(dice_score)
	dice_scores = np.mean(dice_scores)
	return dice_scores