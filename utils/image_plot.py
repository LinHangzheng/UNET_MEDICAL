from PIL import Image
import torch
import os
import numpy as np
COLOR_PALLET = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
                '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
                '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']


RGB_PALLET = torch.Tensor([
    [35, 101, 173],
    [48, 177, 85],
    [76, 134, 178],
    [192, 100, 68],
    [132, 46, 30],
    [234, 36, 46],
    [78, 44, 134],
    [158, 58, 138],
    [217, 194, 216],
    [144, 126, 153],
    [255, 255, 255]
]).to(torch.uint8)


def plot_pred(n_start, num_class, images, labels, preds, save_path):
    preds = torch.argmax(preds,dim=1)
    
    # convert datatype to uint8 for plotting 
    images = (images/torch.max(images)*255).to(torch.uint8)
    preds_RGB = torch.zeros([preds.shape[0],preds.shape[1],preds.shape[2],3],dtype=torch.uint8)
    labels_RGB = torch.zeros([labels.shape[0],labels.shape[1],labels.shape[2],3],dtype=torch.uint8)
    for i in range(num_class):
        preds_RGB[torch.where(preds==i)] = RGB_PALLET[i]
        labels_RGB[torch.where(labels==i)] = RGB_PALLET[i]
    images = torch.moveaxis(images,1,3) 
    
    images = images.to("cpu")
    labels = labels.to("cpu")
    preds = preds.to("cpu")
    
    for i in range(len(images)):
        idx = n_start + i + 1
        
        # only plot the first channel for input images
        image = Image.fromarray(np.array(images[i,:,:,0]))
        label = Image.fromarray(np.array(labels_RGB[i]))
        pred = Image.fromarray(np.array(preds_RGB[i]))
        
        image.save(os.path.join(save_path,f"input_{idx}.jpg"))
        label.save(os.path.join(save_path,f"label_{idx}.jpg"))
        pred.save(os.path.join(save_path,f"pred_{idx}.jpg"))
        
    return 

def plot_entire(IR, label, idx, image_size, net, save_path, plot_entire_pace, num_class=11):
    '''
    Returns: entire prediction [C, H, W]
    '''
    # creat arrays for color plot
    labels_RGB = torch.zeros([label.shape[0],label.shape[1],3],dtype=torch.uint8)
    preds_RGB = torch.zeros([label.shape[0],label.shape[1],3],dtype=torch.uint8)
    preds_mask = torch.zeros([num_class, IR.shape[1],IR.shape[2]],dtype=torch.float32)
    IR = torch.unsqueeze(IR,0)
    IR = IR.type(torch.float32)
    
    # assign color
    for i in range(num_class):
        labels_RGB[torch.where(label==i)] = RGB_PALLET[i]
    
    # plot label
    label = Image.fromarray(np.array(labels_RGB))
    label.save(os.path.join(save_path,f"label_entire_{idx}.jpg"))
   
    for i in range(IR.shape[2]//plot_entire_pace+1):
        h = i*plot_entire_pace
        if h+image_size>IR.shape[2]:
            h = IR.shape[2] - image_size
        for j in range(IR.shape[3]//plot_entire_pace+1):
            w = j*plot_entire_pace
            if w+image_size>IR.shape[3]:
                w = IR.shape[3] - image_size
            with torch.no_grad():
                pred = net(IR[:,:,h:h+image_size,w:w+image_size])
            pred = torch.squeeze(pred,0)
            preds_mask[:,h:h+image_size,w:w+image_size] += pred.cpu()
    preds_mask_post = torch.argmax(preds_mask, dim=0)
    for i in range(num_class):
        preds_RGB[torch.where(preds_mask_post==i)] = RGB_PALLET[i]
    IR = Image.fromarray(np.array(preds_RGB))
    IR.save(os.path.join(save_path,f"preds_entire_{idx}.jpg"))
    return  preds_mask

    
