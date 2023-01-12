import os
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction import image
from patchify import patchify
import numpy as np

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
        self.IR_threshold = params["train_input"]["IR_threshould"]
        self.image_size = params["train_input"]["image_size"]
        self.seed = params["train_input"].get("seed", None)
        self.batch = params["train_input"]["image_size"]
        self.train_test_split = params["train_input"]["train_test_split"]
        self.steps_per_epoch = params["train_input"]["steps_per_epoch"]
        self.large_patch_size = params["train_input"]["large_patch_size"]
        self.patch_h_dim = params["train_input"]["patch_h_dim"]
        self.patch_w_dim = params["train_input"]["patch_h_dim"]
        self.train_patch_step = params["train_input"]["train_patch_step"]
        self.test_patch_step = params["train_input"]["test_patch_step"]
        # IR = np.array(h5py.File(os.path.join(self.data_dir,'IR.mat'), 'r')['X'])
        # IR = self.normolize(IR)
        # label = np.array(h5py.File(os.path.join(self.data_dir,'Class.mat'), 'r')['CL'])
        # IR = np.moveaxis(IR, 0, -1)

        # self.train_IR = None
        # self.test_IR = None
        # self.train_label = None
        # self.test_label = None 
        # self.data_split(IR,label,self.train_test_split)
        
        self.IR_patches = None
        self.label_patches = None
        if mode =="train":
            with open(os.path.join(self.data_dir,'train_IR'),'rb') as f:
                self.train_IR = np.load(f)
            with open(os.path.join(self.data_dir,'train_label'),'rb') as f:
                self.train_label = np.load(f)
                
            self.train_data_preparation(self.train_IR, 
                                self.train_label,
                                self.large_patch_size,
                                self.patch_h_dim,
                                self.patch_w_dim,
                                self.train_patch_step
                                )
            del self.train_IR, self.train_label
        else:
            with open(os.path.join(self.data_dir,'test_IR'),'rb') as f:
                self.test_IR = np.load(f)
            with open(os.path.join(self.data_dir,'test_label'),'rb') as f:
                self.test_label = np.load(f)
            self.test_data_preparation(self.test_IR, 
                                self.test_label,
                                self.image_size,
                                self.IR_threshold,
                                self.test_patch_step)
            del self.test_IR, self.test_label
        
    
    def normolize(self, IR):
        negative_idx = np.where(IR<0)
        IR[negative_idx] = 0
        return IR
    
    def train_data_preparation(self, IR, label, large_patch_size=200, patch_h_dim=15, patch_w_dim=15, patch_step = 20):
        H, W = IR.shape[0], IR.shape[1]
        self.IR_patches, self.label_patches = [],[]
        
        for i in range(patch_h_dim):
            for j in range(patch_w_dim):
                IR_patch_area = IR[i*H//patch_h_dim:(i+1)*H//patch_h_dim,
                                   j*W//patch_w_dim:(j+1)*W//patch_w_dim,
                                   :]
                label_patch_area = label[i*H//patch_h_dim:(i+1)*H//patch_h_dim,
                                         j*W//patch_w_dim:(j+1)*W//patch_w_dim]

                patches = patchify(IR_patch_area[:,:,0], 
                           (large_patch_size,large_patch_size), 
                           step=patch_step)
                patches_idx = np.where(np.mean(patches,axis=(2,3))==np.max(np.mean(patches,axis=(2,3))))
                
                patch_c = []
                for k in range(self.IR_channel_level):
                    patch = patchify(IR_patch_area[:,:,k], 
                                        (large_patch_size,large_patch_size), 
                                        step=patch_step)
                    patch = torch.FloatTensor(patch[patches_idx][0])
                    patch_c.append(patch)
                patch_IR = torch.stack(patch_c)      #[IR_channel_level,H,W]
                # patch_IR = torch.moveaxis(patch_IR, 0,2)
                
                patches_label = patchify(label_patch_area, 
                           (large_patch_size,large_patch_size), 
                           step=patch_step)
                patch_label = patches_label[patches_idx][0].astype(int)
                
                self.IR_patches.append(patch_IR)
                self.label_patches.append(patch_label)

    def test_data_preparation(self, IR, label, patch_size, threshold, patch_step=5):
        H, W = IR.shape[0], IR.shape[1]
        self.IR_patches, self.label_patches = [],[]


        patches = patchify(IR[:,:,0], 
                    (patch_size,patch_size), 
                    step=patch_step)
        patches_idx = np.where(np.mean(patches,axis=(2,3))>threshold)
        
        patch_c = []
        for k in range(self.IR_channel_level):
            patch = patchify(IR[:,:,k], 
                                (patch_size,patch_size), 
                                step=patch_step)
            patch = torch.FloatTensor(patch[patches_idx])
            patch_c.append(patch)
        patch_IR = torch.stack(patch_c)      #[IR_channel_level,N,H,W]
        patch_IR = patch_IR.permute(1,0,2,3)      #[N,IR_channel_level,H,W]
        patches_label = patchify(label, 
                    (patch_size,patch_size), 
                    step=patch_step)
        patch_label = patches_label[patches_idx]
        
        self.IR_patches = patch_IR
        self.label_patches = patch_label.astype(int)
        
        # from matplotlib import pyplot as plt
        # plt.imshow(patch_IR[0][0], interpolation='nearest')
        # plt.show()
        
        # plt.imshow(patch_label[0], interpolation='nearest')
        # plt.show()


    def data_split(self, IR, label, train_test_split):
        split_col = int(IR.shape[1]*train_test_split)
        self.train_IR = IR[:,0:split_col]
        self.test_IR = IR[:,split_col:]
        self.train_label = label[:,:split_col]
        self.test_label = label[:,split_col:]

    
    def __len__(self):
        if self.mode =="train":
            return self.steps_per_epoch*self.batch
        else:
            return len(self.IR_patches)
     
    def __getitem__(self, idx:int):
        if self.mode == "train":
            idx = idx%(self.patch_h_dim*self.patch_w_dim)
            h,w = torch.randint(high=self.large_patch_size-self.image_size-1,size=(2,))
            return self.IR_patches[idx][:,h:h+self.image_size,w:w+self.image_size], self.label_patches[idx][h:h+self.image_size,w:w+self.image_size] 
        else:
            return self.IR_patches[idx], self.label_patches[idx]
