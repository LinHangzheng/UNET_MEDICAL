import os
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction import image
from patchify import patchify
import numpy as np
from PIL import Image
from glob import glob
import torchvision.transforms.functional as TF
class IRDataset(Dataset):
    def __init__(self, 
        params=None, 
        mode='train'
    ):
        self.mode = mode
        self.params = params
        self.data_dir = params["train_input"]["dataset_path"]
        self.IR_channel_level = params["train_input"]["IR_channel_level"]
        self.num_classes = params["train_input"]["num_classes"]
        self.image_size = params["train_input"]["image_size"]
        self.seed = params["train_input"].get("seed", None)
        self.train_test_split = params["train_input"]["train_test_split"]
        self.patch_h_dim = params["train_input"]["train_patch_h_dim"] if mode =='train' else params["train_input"]["test_patch_h_dim"]
        self.patch_w_dim = params["train_input"]["train_patch_w_dim"] if mode =='train' else params["train_input"]["test_patch_w_dim"]
        self.patch_step = params["train_input"]["patch_step"]
        self.augment_data = params["train_input"]["augment_data"]
        self.noise_variance = params["train_input"]["noise_variance"]
        
        if mode == 'train':
            self.IR = sorted(glob(os.path.join(self.data_dir,'train','IR/*')))
            self.label = sorted(glob(os.path.join(self.data_dir,'train','label/*')))
        else:
            self.IR = sorted(glob(os.path.join(self.data_dir,'test','IR/*')))
            self.label = sorted(glob(os.path.join(self.data_dir,'test','label/*')))
        
        self.IR_patches = torch.from_numpy(np.array([np.load(path) for path in self.IR]).astype(np.float32))
        self.label_patches = torch.from_numpy(np.array([np.load(path) for path in self.label]).astype(np.int64))
        self.large_patch_size = self.IR_patches[0].shape[0]
        self.IR_patches = torch.moveaxis(self.IR_patches, 3,1)
        
        
        self.data_augmentation()
        self.label_patches = torch.unsqueeze(self.label_patches, dim=1)
        # for i in range(self.IR_patches.shape[0]):
        #     angle = 360*torch.rand(1).item()
        #     IR = TF.rotate(self.IR_patches[i], angle)
        #     label = TF.rotate(self.label_patches[i], angle)
        #     label = torch.squeeze(label, dim=0)/6*255
        #     label = np.array(label).astype(np.uint8)
        #     img = Image.fromarray(label)
            
        #     img.save(f"{self.mode}_label_{i}.jpg")
        #     # IR = self.IR_patches[i][4] + (0.0001**0.5)*torch.randn(self.IR_patches[i][4].shape)
        #     # img = Image.fromarray(np.array((IR-torch.min(IR))/torch.max(IR)*255,dtype=np.uint8))
        #     # img.save(f"{self.mode}_IR_{i}.jpg")
        #     img = Image.fromarray(np.array((IR[4]-torch.min(IR[4]))/torch.max(IR[4])*255).astype(np.uint8))
        #     img.save(f"{self.mode}_IR_{i}_ori.jpg")
        

    def data_augmentation(self):
        if not self.augment_data:
            return 
        IR_flipx = torch.flip(self.IR_patches,[2])
        label_flipx = torch.flip(self.label_patches,[1])
        self.IR_patches = torch.concat([self.IR_patches,IR_flipx])
        self.label_patches = torch.concat([self.label_patches,label_flipx])
        
        
    def __len__(self):
        if self.mode =="train":
            return self.IR_patches.shape[0]
        else:
            return self.IR_patches.shape[0]*4
        
    def __getitem__(self, idx:int):
        if self.mode == "train":
            h,w = torch.randint(high=self.large_patch_size-self.image_size-1,size=(2,))
            IR = self.IR_patches[idx][:,h:h+self.image_size,w:w+self.image_size]
            # IR = IR + (self.noise_variance**0.5)*torch.randn(IR.shape)
            label = self.label_patches[idx][:,h:h+self.image_size,w:w+self.image_size] 
            
            angle = 360*torch.rand(1).item()
            IR = TF.rotate(IR, angle)
            label = TF.rotate(label, angle)
            label = torch.squeeze(label, dim=0)
        else:
            round = idx //self.IR_patches.shape[0]
            idx = idx % self.IR_patches.shape[0] 
            h = (self.large_patch_size-self.image_size)//2*(round//2)
            w = (self.large_patch_size-self.image_size)//2*(round%2)
            IR = self.IR_patches[idx][:,h:h+self.image_size,w:w+self.image_size]
            label = self.label_patches[idx][:,h:h+self.image_size,w:w+self.image_size] 
            label = torch.squeeze(label, dim=0)
        return IR, label
            
        
