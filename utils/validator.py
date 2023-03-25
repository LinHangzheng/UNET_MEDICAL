# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch 
import os

from input import IRDatasetProcessor
from .metric import compute_acu, compute_auc, plot_roc, compute_dice
from .image_plot import plot_pred, plot_entire
from .loss import CombinedLoss
from einops import rearrange
import time
import numpy as np
#from tqdm import tqdm
class Validator(object):
    """Geometric validation; sample 3D points for distance/occupancy metrics."""

    def __init__(self, params, device, net):
        self.params = params
        self.num_class = params["train_input"]["num_classes"]
        self.valid_only = params["runconfig"]["valid_only"]
        self.batch_size = params["eval_input"]["batch_size"]
        self.plot_path = params["eval_input"]["plot_path"]
        self.threshold = params["eval_input"]["threshold"]
        self.plot_entire_idx = params["eval_input"]["plot_entire_idx"]
        self.plot_roc = params["eval_input"]["plot_roc"]
        self.image_shape = params["train_input"]["image_shape"]
        self.plot_entire_pace = params["eval_input"]["plot_entire_pace"]
        self.device = device
        self.net = net
        self.set_dataset()

    def set_dataset(self):
        self.DatasetProcessor = IRDatasetProcessor(self.params)
        self.val_data_loader = self.DatasetProcessor.create_dataloader(
                                    is_training=False)


    def validate(self, epoch):
        """Geometric validation; sample surface points."""
        val_dict = {}
        total = 0
        # Uniform points metrics
        self.net.eval() 
        preds_total = []
        labels_total = []
        for n_iter, data in enumerate(self.val_data_loader):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)
            with torch.no_grad():
                preds = self.net(images)
            if self.valid_only: 
                plot_pred(n_iter*self.batch_size,self.num_class,images,labels,preds,self.plot_path)
            preds_total.append(preds)
            labels_total.append(labels)
            total += images.shape[0]
        preds = torch.cat(preds_total,dim=0)
        labels = torch.cat(labels_total,dim=0)
        val_dict['DICE'] = compute_dice(preds.softmax(dim=1), labels,self.num_class)
        preds = rearrange(preds, 'b c h w -> (b h w) c')
        val_dict['AUC'] = [compute_auc(preds, labels, self.num_class,thresholds=self.threshold, device=self.device)]
        val_dict['ACU'] = compute_acu(preds,labels,self.num_class,True)
        val_dict['AUC'] = torch.stack(val_dict['AUC'])
        val_dict['AUC'] = torch.sum(val_dict['AUC'],axis=0)
        
        
        for i in range(self.num_class):
            val_dict[f'AUC_{i+1}'] = val_dict['AUC'][i]
        val_dict['AUC'] = torch.mean(val_dict['AUC'])
        acu = []
        time_list = []
        if self.valid_only:
            if self.plot_entire_idx is not None:
                for i in range(self.plot_entire_idx):
                    IR, label = self.val_data_loader.dataset.get_entire_image(i)
                    start = time.time()
                    preds_IR = plot_entire(IR, label, i, self.image_shape[0], self.net, self.plot_path, self.plot_entire_pace)
                    end = time.time()
                    time_list.append(end-start)
                    preds_IR = rearrange(preds_IR, 'h w -> (h w)')
                    
                    acu.append(compute_acu(preds_IR, label,self.num_class,True))
                print(f"entire acu: {np.mean(acu)}")
                print(f"entire time: {np.mean(time_list)}")
            print(f"enter valid only: AUC={val_dict['AUC']}")
            if self.plot_roc:
                plot_roc(preds,labels,self.num_class,os.path.join(self.plot_path,"ROC_figure.jpg"))
            with open(os.path.join(self.plot_path,'result.txt'),'w') as f:
                for i in range(self.num_class):
                    auc = val_dict[f'AUC_{i+1}']
                    f.write(f"{auc}\n")
                f.write(f"{val_dict['AUC']}")
                f.write("\n\n")
                f.write(f"{val_dict['DICE']}")
        return val_dict

    
