import torch
from torchmetrics.classification import MulticlassAUROC, MulticlassROC
import matplotlib.pyplot as plt
import numpy as np
import time
RGB_PALLET = np.array([
    [35, 101, 173, 255],
    [48, 177, 85, 255],
    [76, 134, 178, 255],
    [192, 100, 68, 255],
    [132, 46, 30, 255],
    [234, 36, 46, 255],
    [78, 44, 134, 255],
    [158, 58, 138, 255],
    [217, 194, 216, 255],
    [144, 126, 153, 255],
    [255, 255, 255, 255]
])/255


def remove_background(pred, labels, background):
    ''' 
    remove the -1 and background from both predictions and labels
    Args:
        pred: [N,...] prediction (N would be any shape)
        labels: [N] labels (N would be any shape)
        background: the background index
        
    Returns:
        pred: the new prediction without -1 and background on labels
        labels: the new labels without -1 and background
    '''
    pos = torch.where((labels!=-1) & (labels!=background))
    pred = pred[pos]
    labels = labels[pos]
    return pred, labels

def compute_acu(pred, labels):
    ''' 
    Calculate accuracy 
    Args:
        pred: [(B*H*W)] 1D prediction
        labels: [(B*H*W)] 1D labels
        
    Returns: 
        acu value: would be total_correct/total
    '''
    total_correct = torch.where(pred==labels)[0].shape[0]
    total = labels.shape[0]
    acu_total = 100.*total_correct/(total+1e-5)
    return torch.tensor(acu_total)\

def compute_auc(pred, labels, num_classes, average=None, thresholds=None, device=None):
    ''' 
    Calculate AUC (Area Under the Curve) 
    Args:
        pred: [(B*H*W), C] 2D prediction
        labels: [(B*H*W)] 1D labels
        
    Returns:
        AUC value
    '''
    mc_auroc = MulticlassAUROC(
                num_classes=num_classes, 
                average=average, 
                thresholds=thresholds
                ).to(device)
    return mc_auroc(pred, labels)

def compute_dice(pred, label , num_classes, epsilon=1e-6):
    """
    Compute the Dice metric for multi-class segmentation between label and pred.

    Args:
        pred: predicted tensor with shape [batch_size, num_classes, H, W]
        label: ground truth tensor with shape [batch_size, H, W]
        num_classes: number of classes in the segmentation task (excluding the background class)
        epsilon: a small constant to avoid division by zero

    Returns:
        Dice metric for multi-class segmentation
    """
    total_dice_metric = 0.

    for i in range(num_classes):
        # create binary masks for each class
        label_class = (label == i).float()
        pred_class = pred[:, i, :, :]

        # compute dice metric for each class
        intersection = torch.sum(label_class * pred_class, dim=(1,2))
        label_volume = torch.sum(label_class, dim=(1,2))
        pred_volume = torch.sum(pred_class, dim=(1,2))
        union = label_volume + pred_volume + epsilon
        dice = (2. * intersection + epsilon) / union
        dice_metric = torch.mean(dice)
        total_dice_metric += dice_metric

    return total_dice_metric / num_classes

def compute_dice_2D(preds, labels):
    """
    Compute the Dice metric for multi-class segmentation between label and pred.

    Args:
        pred: predicted tensor with shape [N, C]
        label: ground truth tensor with shape [N]
        epsilon: a small constant to avoid division by zero

    Returns:
        Dice metric for multi-class segmentation
    """
    assert preds.shape[0] == labels.shape[0], "Mismatch in the number of samples"
    
    N, C = preds.shape
    dice_scores = torch.zeros(C) 
    one_hot_labels = torch.eye(C)[labels.to(torch.long)]
    for c in range(C):
        pred_c = preds[:, c]
        label_c = one_hot_labels[:, c]
        
        intersection = torch.sum(pred_c * label_c)
        union = torch.sum(pred_c) + torch.sum(label_c)
        
        if union == 0:
            dice_scores[c] = 1.0
        else:
            dice_scores[c] = 2 * intersection / union
    
    return dice_scores

def plot_roc(pre, labels, num_classes, save_path, thresholds=None):
    y = labels.view(-1)
    metric = MulticlassROC(num_classes=num_classes, thresholds=thresholds)
    fpr, tpr, thresholds = metric(pre, y)
    plt.figure()
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    print("start plotting")
    for i in range(num_classes):
        plt.plot(fpr[i].cpu(),tpr[i].cpu(),color=RGB_PALLET[i],label=f"ROC_{i+1}")
    plt.title('ROC')
    plt.xlabel("True Postive Rate")
    plt.ylabel("False Postive Rate")
    
    plt.savefig(save_path)
    
