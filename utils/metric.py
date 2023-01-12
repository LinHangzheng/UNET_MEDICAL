import torch

def compute_acu(pre, labels,num_classes):
    """Intersection over Union.
    
    Args:
        dist_gt (torch.Tensor): Groundtruth signed distances
        dist_pr (torch.Tensor): Predicted signed distances
    """
    x = pre.moveaxis(1,3)
    x = x.reshape(-1,num_classes)
    x = torch.argmax(x,dim=1)
    y = labels.view(-1)
    correct = torch.where(x==y)[0].shape[0]
    total = x.shape[0]
    ACU = correct / total
    return 100. * ACU