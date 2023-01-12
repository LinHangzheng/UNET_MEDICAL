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
    total = x.shape[0]
    ret = []
    
    for c in range(num_classes):
        TP = torch.where((x==y) & (y==c))[0].shape[0]
        TN = torch.where((x!=c) & (y!=c))[0].shape[0]
        auc = 100. * (TP+TN)/total 
        ret.append(auc)    
        
    total_correct = torch.where(x==y)[0].shape[0]
    auc_total = 100.*total_correct/total
    ret.append(auc_total)
    return ret