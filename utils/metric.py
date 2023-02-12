import torch
from torchmetrics.classification import MulticlassAUROC, MulticlassROC
import matplotlib.pyplot as plt

COLOR_MAPS = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
              'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
              'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
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
    metric = MulticlassROC(num_classes=num_classes, thresholds=thresholds)
    fpr, tpr, thresholds = metric(pre, labels)
    fig = plt.figure()
    for i in range(num_classes):
        plt.plot(fpr,tpr,color=COLOR_MAPS[i])
    plt.title('ROC')
    plt.title()
    plt.savefig('ROC_figure.jpg')
    
