import os
import torch
from torchvision import transforms
from glob import glob
from torchvision.datasets import VisionDataset
from .image_processing import cenInd
import torch.distributed as dist
import nibabel as nib
import numpy as np

from torch.utils.data.distributed import DistributedSampler
# from PIL import Image
from .preprocessing_utils import (
    adjust_brightness_transform,
)

class BraTsDataset(VisionDataset):
    def __init__(
        self, 
        root, 
        split='train',
        crop_sz=[160,192,128],
        transforms=None,
        transform=None,
        target_transform=None
    ):
        super(BraTsDataset, self).__init__(
            root, transforms, transform, target_transform
        )
        self.split = split
        self.root = root
        self.fileName = sorted(glob(os.path.join(self.root,split,'Bra*')))
        self.crop_sz  = crop_sz

    def __len__(self):
        return 10000000

    def __getitem__(self, index:int):
        index = index % len(self.fileName)
        fileName_indv         = self.fileName[index]
        
        ###============= Data Loading ===================### 
        image_flair           = np.expand_dims(nib.load(glob(fileName_indv + '/*flair.nii.gz')[0]).get_fdata(),axis=0)
        image_t1              = np.expand_dims(nib.load(glob(fileName_indv + '/*_t1.nii.gz')[0]).get_fdata(),axis=0)
        image_t1ce            = np.expand_dims(nib.load(glob(fileName_indv + '/*_t1ce.nii.gz')[0]).get_fdata(),axis=0)
        image_t2              = np.expand_dims(nib.load(glob(fileName_indv + '/*_t2.nii.gz')[0]).get_fdata(),axis=0)        
        images                = np.concatenate((image_flair, image_t1, image_t1ce, image_t2), axis=0)
        
        labels                = nib.load(glob(fileName_indv + '/*_seg.nii.gz')[0]).get_fdata()
        
        ###======= prep-processing labels ===============### 
        ncr                   = labels == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
        ed                    = labels == 2  # Peritumoral Edema (ED)
        et                    = labels == 4  # GD-enhancing Tumor (ET)        
        labels                = np.array([ncr, ed, et], dtype=np.uint8)
        
        ###============= crop images ===================###
        [Ny,Nx,Nz]            = images.shape[1:]
        images                = images[:,cenInd(Ny,self.crop_sz[0])[0]:cenInd(Ny,self.crop_sz[0])[-1]+1,cenInd(Nx,self.crop_sz[1])[0]:cenInd(Nx,self.crop_sz[1])[-1]+1,cenInd(Nz,self.crop_sz[2])[0]:cenInd(Nz,self.crop_sz[2])[-1]+1]
        labels                = labels[:,cenInd(Ny,self.crop_sz[0])[0]:cenInd(Ny,self.crop_sz[0])[-1]+1,cenInd(Nx,self.crop_sz[1])[0]:cenInd(Nx,self.crop_sz[1])[-1]+1,cenInd(Nz,self.crop_sz[2])[0]:cenInd(Nz,self.crop_sz[2])[-1]+1]
        
        ###============= To Torch ===================###
        images                = torch.from_numpy(images).float()
        labels                = torch.from_numpy(labels).float()
        
        ###============= normalization ===================###
        
        # calculate the mean and std
        mean, std             = images.mean([1,2,3]), images.std([1,2,3])
        
        # normalization
        images                = (images - mean.view([mean.shape[0],1,1,1])) / std.view([std.shape[0],1,1,1])
        
        return images, labels
        
class BraTsDatasetProcessor(VisionDataset):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.data_dir = params["train_input"]["data_dir"]
        self.image_shape = params["train_input"]["image_shape"]  # of format (H, W, D, C)

        self.shuffle_seed = params["train_input"].get("shuffle_seed", None)
        if self.shuffle_seed:
            torch.manual_seed(self.shuffle_seed)

        self.augment_data = params["train_input"].get("augment_data", True) 
        self.shuffle = params["train_input"].get("shuffle", True)

        # Multi-processing params.
        self.num_workers = params["train_input"].get("num_workers", 0)
        self.drop_last = params["train_input"].get("drop_last", True)
        self.prefetch_factor = params["train_input"].get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)
        self.world_size = params["runconfig"]["world_size"]
        self.mp_type = torch.float32
        
        # default is that each activation worker sends `num_workers`
        # batches so total batch_size * num_act_workers * num_pytorch_workers samples

        # Using Faster Dataloader for mapstyle dataset.

            
    def create_dataset(self, is_training):
        split = "train" if is_training else "val"
        dataset = BraTsDataset(
            root=self.data_dir,
            split=split,
            crop_sz=self.image_shape[:3],
            transforms=self.transform_image_and_mask,
        )
        return dataset

    def create_dataloader(self, is_training=False, rank=0):
        self.is_training = is_training
        batch_size = self.params["train_input"]["batch_size"] if is_training else self.params["eval_input"]["batch_size"]
        dataset = self.create_dataset(is_training)
        generator_fn = torch.Generator(device='cpu')
        if self.shuffle_seed is not None:
            generator_fn.manual_seed(self.shuffle_seed)

        data_sampler = torch.utils.data.SequentialSampler(dataset)

        if self.world_size > 1 and is_training:
            data_sampler = DistributedSampler(dataset,
                        num_replicas=self.world_size, 
                        rank=rank, 
                        shuffle=self.shuffle, 
                        drop_last=self.drop_last)
        elif self.shuffle and is_training:
            seed = self.shuffle_seed + dist.get_rank()
            generator_fn.manual_seed(seed)
            data_sampler = torch.utils.data.RandomSampler(
                dataset, generator=generator_fn
            )
        else:
            data_sampler = torch.utils.data.SequentialSampler(dataset)
            
        dataloader_fn = torch.utils.data.DataLoader
        print("-- Using torch.utils.data.DataLoader -- ")

        if self.num_workers:
            dataloader = dataloader_fn(
                dataset,
                batch_size=batch_size,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                drop_last=self.drop_last,
                generator=generator_fn,
                sampler=data_sampler,
            )
        else:
            dataloader = dataloader_fn(
                dataset,
                batch_size=batch_size,
                drop_last=self.drop_last,
                generator=generator_fn,
                sampler=data_sampler,
            )
        return dataloader

    def transform_image_and_mask(self, image, mask):
        # image: [C, H, W]
        if self.augment_data and self.is_training:
            delta = torch.rand(1) * 0.5
            augment_transform_image = self.get_augment_transforms(
                do_random_brightness=True,
                delta = delta
            )
            augment_transform_mask = self.get_augment_transforms(
                do_random_brightness=False,
                delta = delta
            )

            image = augment_transform_image(image)
            mask = augment_transform_mask(mask)

        # Handle dtypes and mask shapes based on `loss_type`
        # and `mixed_precsion`


        image = image.type(self.mp_type)
        mask = mask.type(torch.int64)
        return image, mask

    def get_augment_transforms(
        self, do_random_brightness, delta
    ):
        augment_transforms_list = []
        if self.is_training:
            if do_random_brightness:
                brightness_transform = transforms.Lambda(
                    lambda x: adjust_brightness_transform(x, p=0.5, delta=delta)
                )
                augment_transforms_list.append(brightness_transform)
            
        return transforms.Compose(augment_transforms_list)
