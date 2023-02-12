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

import numpy as np

from input import IRDatasetProcessor
from .metric import compute_acu
from einops import rearrange
class Validator(object):
    """Geometric validation; sample 3D points for distance/occupancy metrics."""

    def __init__(self, params, device, net):
        self.params = params
        self.num_class = params["train_input"]["num_classes"]
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
        val_dict['ACU'] = []
        
        total = 0
        # Uniform points metrics
        
        for n_iter, data in enumerate(self.val_data_loader):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)
            
            preds = self.net(images)
            preds = rearrange(preds, 'b c h w -> (b h w) c')
            val_dict['ACU'] += [compute_acu(preds, labels, self.num_class)]*images.shape[0]
            total += images.shape[0]
        
        val_dict['ACU'] = np.sum(val_dict['ACU'],axis=0)/total
        for i in range(1, self.num_class+1):
            val_dict[f'ACU_{i}'] = val_dict['ACU'][i-1]
        val_dict['ACU'] = val_dict['ACU'][-1]
        return val_dict

    
