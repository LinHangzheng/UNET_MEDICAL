import torch.nn as nn
import torch

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, epsilon=1e-6):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.epsilon = epsilon

    def forward(self, pred, target):
        dice_loss = CombinedLoss.soft_dice_loss_multiclass(pred.softmax(dim=1), target, pred.shape[1], self.epsilon)
        ce = self.cross_entropy_loss(pred, target)
        return self.weight_dice * dice_loss + self.weight_ce * ce, 1-dice_loss, ce
        
    @staticmethod
    def soft_dice_loss_multiclass(pred, target, num_classes, epsilon=1e-6):
        """
        Compute the Soft Dice Loss for multi-class segmentation between target and pred.

        Args:
            target: ground truth tensor with shape [batch_size, num_classes, H, W]
            pred: predicted tensor with shape [batch_size, num_classes, H, W]
            num_classes: number of classes in the segmentation task
            epsilon: a small constant to avoid division by zero

        Returns:
            Soft Dice Loss for multi-class segmentation
        """
        total_soft_dice_loss = 0.

        for i in range(1,num_classes):
            # create binary masks for each class
            target_class = (target == i).float()
            pred_class = pred[:, i, :, :]

            # compute soft dice loss for each class
            intersection = torch.sum(target_class * pred_class, dim=(1,2))
            target_volume = torch.sum(target_class, dim=(1,2))
            pred_volume = torch.sum(pred_class, dim=(1,2))
            union = target_volume + pred_volume + epsilon
            soft_dice = (2. * intersection + epsilon) / union
            soft_dice_loss = 1. - torch.mean(soft_dice)
            total_soft_dice_loss += soft_dice_loss

        return total_soft_dice_loss / num_classes



    def cross_entropy_loss(self, pred, target):
        loss = nn.CrossEntropyLoss()
        n_class = pred.shape[1]
        idx = torch.where(target<=n_class)
        target = target[idx]
        pred = torch.moveaxis(pred,1,-1)
        pred = pred[idx]
        return loss(pred,target)
