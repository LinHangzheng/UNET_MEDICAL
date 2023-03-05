import torch.nn as nn
import torch.nn.functional as F
import torch

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.softmax(dim=1)
        dice_loss, dice = self.dice_loss(pred, target, self.smooth)
        ce = self.cross_entropy_loss(pred, target)
        return self.weight_dice * dice_loss + self.weight_ce * ce, dice, ce
        
    @staticmethod
    def dice_loss(self, pred, target, smooth):
        num_classes = pred.shape[1]
        dice = torch.zeros(num_classes, device=pred.device)
        for c in range(num_classes):
            p = pred[:,c,:,:]
            t = (target == c).float()
            intersection = (p * t).sum()
            dice[c] = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
        return 1.0 - dice.mean(), dice

    def cross_entropy_loss(self, pred, target):
        return F.cross_entropy(pred, target)