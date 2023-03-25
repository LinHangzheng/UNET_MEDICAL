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

def compute_acu(pre, labels, num_classes, only_total=False):
    x = pre
    if len(x.shape) == 2:
        x = torch.argmax(x,dim=1)
    y = labels.contiguous().view(-1)
    total = x.shape[0]
    
    total_correct = torch.where(x==y)[0].shape[0]
    auc_total = 100.*total_correct/total
    if only_total:
        return auc_total
    
    ret = []
    for c in range(num_classes):
        TP = torch.where((x==y) & (y==c))[0].shape[0]
        TN = torch.where((x!=c) & (y!=c))[0].shape[0]
        auc = 100. * (TP+TN)/total 
        ret.append(auc)    
    
    
    ret.append(auc_total)
    ret = torch.Tensor(ret)
    return ret

def compute_auc(pre, labels, num_classes, average=None, thresholds=None, device=None):
    y = labels.view(-1)
    mc_auroc = MulticlassAUROC(num_classes=num_classes, average=average, thresholds=thresholds).to(device)
    return mc_auroc(pre, y)

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
    
