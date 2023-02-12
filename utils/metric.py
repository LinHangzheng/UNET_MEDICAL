import torch

def compute_acu(pre, labels, num_classes, only_total=False):
    x = pre
    x = torch.argmax(x,dim=1)
    y = labels.view(-1)
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
    
    # ROC curve
        
    
    ret.append(auc_total)
    return ret