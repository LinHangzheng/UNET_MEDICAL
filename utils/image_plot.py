from PIL import Image
import torch
import os
import numpy as np


def plot_pred(n_start, num_class, images, labels, preds, save_path):
    preds = torch.argmax(preds,dim=1)
    
    # convert datatype to uint8 for plotting 
    images = (images/torch.max(images)*255).to(torch.uint8)
    labels = (labels/num_class*255).to(torch.uint8)
    preds = (preds/torch.max(preds)*255).to(torch.uint8)
    
    images = torch.moveaxis(images,1,3) 
    
    images = images.to("cpu")
    labels = labels.to("cpu")
    preds = preds.to("cpu")
    
    for i in range(len(images)):
        idx = n_start + i + 1
        
        # only plot the first channel for input images
        image = Image.fromarray(np.array(images[i,:,:,0]))
        label = Image.fromarray(np.array(labels[i]))
        pred = Image.fromarray(np.array(preds[i]))
        
        image.save(os.path.join(save_path,f"input_{idx}.jpg"))
        label.save(os.path.join(save_path,f"label_{idx}.jpg"))
        pred.save(os.path.join(save_path,f"pred_{idx}.jpg"))
        
    return 