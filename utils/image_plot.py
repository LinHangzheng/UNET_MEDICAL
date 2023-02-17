from PIL import Image
import torch
import os
import numpy as np

COLOR_PALLET = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
                '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
                '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']

RGB_PALLET = torch.Tensor(
[[215, 39, 79],
          [85, 167, 98],
          [229, 206, 56],
          [28, 142, 175],
          [214, 131, 81],
          [122, 52, 171],
          [119, 212, 212],
          [210, 67, 197],
          [201, 225, 76],
          [228, 186, 202],
          [0, 117, 117],
          [205, 191, 228],
          [155, 105, 48],
          [229, 223, 191],
          [108, 21, 21],
          [155, 225, 209],
          [122, 122, 21],
          [229, 193, 172],
          [26, 26, 122],
          [122, 122, 122]]).to(torch.uint8)

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