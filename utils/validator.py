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

import os
import sys
import itertools as it

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from input import IRDataset
from .metric import compute_acu
class Validator(object):
    """Geometric validation; sample 3D points for distance/occupancy metrics."""

    def __init__(self, params, device, net):
        self.params = params
        self.device = device
        self.net = net
        self.set_dataset()

    def set_dataset(self):
        self.val_dataset = IRDataset(self.params, "eval")
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.params["train_input"]["eval_batch_size"], 
                                            shuffle=False, pin_memory=True, num_workers=4)


    def validate(self, epoch):
        """Geometric validation; sample surface points."""
        num_classes = self.params["train_input"]["num_classes"]
        val_dict = {}
        val_dict['ACU'] = []
        
        # Uniform points metrics
        for n_iter, data in enumerate(self.val_data_loader):
            images = data[0].to(self.device)
            labels = data[1].to(self.device)

            pred = self.net(images)
            val_dict['ACU'] += [compute_acu(pred, labels, num_classes)]
        val_dict['ACU'] = np.mean(val_dict['ACU'],axis=0)
        for i in range(1, num_classes+1):
            val_dict[f'ACU_{i}'] = val_dict['ACU'][i-1]
        val_dict['ACU'] = val_dict['ACU'][-1]
        return val_dict

    