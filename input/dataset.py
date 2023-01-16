import os
import h5py
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction import image
from patchify import patchify
import numpy as np
from PIL import Image
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
        self.large_patch_size = params["train_input"]["large_patch_size"]
        self.patch_h_dim = params["train_input"]["train_patch_h_dim"] if mode =='train' else params["train_input"]["test_patch_h_dim"]
        self.patch_w_dim = params["train_input"]["train_patch_w_dim"] if mode =='train' else params["train_input"]["test_patch_w_dim"]
        self.patch_step = params["train_input"]["patch_step"]
        self.augment_data = params["train_input"]["augment_data"]
        self.noise_variance = params["train_input"]["noise_variance"]
        
        self.IR_patches = None
        self.label_patches = None
        if mode =="train":
            with open(os.path.join(self.data_dir,'train_IR'),'rb') as f1, \
                open(os.path.join(self.data_dir,'train_label'),'rb') as f2:
                self.IR = np.load(f1)
                self.label = np.load(f2)
        else:
            with open(os.path.join(self.data_dir,'test_IR'),'rb') as f1, \
                open(os.path.join(self.data_dir,'test_label'),'rb') as f2:
                self.IR = np.load(f1)
                self.label = np.load(f2)
                
        self.data_preparation(self.IR, 
                            self.label,
                            self.large_patch_size,
                            self.patch_h_dim,
                            self.patch_w_dim,
                            self.patch_step
                            )
        del self.IR, self.label
        self.data_augmentation()
        for i in range(self.IR_patches.shape[0]):
            img = Image.fromarray(np.array(self.label_patches[i]/6*255,dtype=np.uint8))
            
            img.save(f"{self.mode}_label_{i}.jpg")
            IR = self.IR_patches[i][4] + (0.0001**0.5)*torch.randn(self.IR_patches[i][4].shape)
            img = Image.fromarray(np.array((IR-torch.min(IR))/torch.max(IR)*255,dtype=np.uint8))
            img.save(f"{self.mode}_IR_{i}.jpg")
            img = Image.fromarray(np.array((self.IR_patches[i][4]-torch.min(self.IR_patches[i][4]))/torch.max(self.IR_patches[i][4])*255,dtype=np.uint8))
            img.save(f"{self.mode}_IR_{i}_ori.jpg")
            
    def normolize(self, IR):
        negative_idx = np.where(IR<0)
        IR[negative_idx] = 0
        return IR
    
    def data_preparation(self, IR, label, large_patch_size=200, patch_h_dim=15, patch_w_dim=15, patch_step = 20):
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

        self.IR_patches = torch.stack(self.IR_patches)
        self.label_patches = torch.from_numpy(np.array(self.label_patches))
                

    def data_split(self, IR, label, train_test_split):
        split_col = int(IR.shape[1]*train_test_split)
        self.train_IR = IR[:,0:split_col]
        self.test_IR = IR[:,split_col:]
        self.train_label = label[:,:split_col]
        self.test_label = label[:,split_col:]

    def data_augmentation(self):
        if not self.augment_data:
            return 
        IR_rot90 = torch.rot90(self.IR_patches,1,[2,3])
        IR_rot180 = torch.rot90(IR_rot90,1,[2,3])
        IR_rot270 = torch.rot90(IR_rot180,1,[2,3])
        IR_flipx = torch.flip(self.IR_patches,[2])
        IR_flipy = torch.flip(self.IR_patches,[3])
        label_rot90 = torch.rot90(self.label_patches,1,[1,2])
        label_rot180 = torch.rot90(label_rot90,1,[1,2])
        label_rot270 = torch.rot90(label_rot180,1,[1,2])
        label_flipx = torch.flip(self.label_patches,[1])
        label_flipy = torch.flip(self.label_patches,[2])
        self.IR_patches = torch.concat([self.IR_patches,IR_rot90,IR_rot180,IR_rot270,IR_flipx,IR_flipy])
        self.label_patches = torch.concat([self.label_patches,label_rot90,label_rot180,label_rot270,label_flipx,label_flipy])
        
        
    def __len__(self):
        if self.mode =="train":
            return self.IR_patches.shape[0]
        else:
            return self.IR_patches.shape[0]*4
        
    def __getitem__(self, idx:int):
        if self.mode == "train":
            h,w = torch.randint(high=self.large_patch_size-self.image_size-1,size=(2,))
            IR = self.IR_patches[idx][:,h:h+self.image_size,w:w+self.image_size]
            IR = IR + (self.noise_variance**0.5)*torch.randn(IR.shape)
            label = self.label_patches[idx][h:h+self.image_size,w:w+self.image_size] 
        else:
            round = idx //self.IR_patches.shape[0]
            idx = idx % self.IR_patches.shape[0] 
            h,w = self.image_size*(round//2),self.image_size*(round%2)
            IR = self.IR_patches[idx][:,h:h+self.image_size,w:w+self.image_size]
            label = self.label_patches[idx][h:h+self.image_size,w:w+self.image_size] 
        return IR, label
            
        
