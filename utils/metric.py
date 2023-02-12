import torch
from torchmetrics.classification import MulticlassAUROC, MulticlassROC
import matplotlib.pyplot as plt

cmap = plt.get_cmap('plasma')
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
    
    
    ret.append(auc_total)
    ret = torch.Tensor(ret)
    return ret

def compute_auc(pre, labels, num_classes, average=None, thresholds=None):
    y = labels.view(-1)
    mc_auroc = MulticlassAUROC(num_classes=num_classes, average=average, thresholds=thresholds)
    return mc_auroc(pre, y)

def plot_roc(pre, labels, num_classes, save_path, thresholds=None):
    slicedCM = cmap(torch.linspace(0, 1, num_classes)) 
    y = labels.view(-1)
    metric = MulticlassROC(num_classes=num_classes, thresholds=thresholds)
    fpr, tpr, thresholds = metric(pre, y)
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i].cpu(),tpr[i].cpu(),color=slicedCM[i],label=f"ROC_{i+1}")
    plt.title('ROC')
    plt.title()
    plt.xlabel("True Postive Rate")
    plt.ylabel("False Postive Rate")
    
    plt.savefig(save_path)
    
