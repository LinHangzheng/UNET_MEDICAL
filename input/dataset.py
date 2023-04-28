import os
import torch
from torchvision import transforms
from glob import glob
from torchvision.datasets import VisionDataset
import torch.distributed as dist
import numpy as np
from torch.utils.data.distributed import DistributedSampler
# from PIL import Image
from .preprocessing_utils import (
    adjust_brightness_transform,
    rotation_90_transform,
)

class IRDataset(VisionDataset):
    def __init__(
        self, 
        root, 
        split='train',
        input_dim = 3,
        transforms=None,
        transform=None,
        target_transform=None
    ):
        super(IRDataset, self).__init__(
            root, transforms, transform, target_transform
        )
        self.split = split
        self.root = root
        self.input_dim = input_dim
        self.IR = sorted(glob(os.path.join(self.root,split,'IR/*.npy')))
        self.label = sorted(glob(os.path.join(self.root,split,'CL/*.npy')))
        
        # get the entire validation images
        self.channel_map = [5,9,2,8,4,7,1,3,6,0]
    
    def __len__(self):
        if self.split == 'train':
            return 12000000
        else:       
            return len(self.IR)
        #return len(self.IR)

    def __getitem__(self, idx:int):
        idx = idx%len(self.IR)
        patch = torch.from_numpy(np.load(self.IR[idx])[self.channel_map[:self.input_dim],:,:])
        label = torch.from_numpy(np.load(self.label[idx]))

        if self.transforms is not None:
            patch, label = self.transforms(patch, label)
        
                
        # for i in range(len(self.channel_map)):
        #     plot = np.array(patch[i,:,:])
        #     plot = np.moveaxis(plot,0,-1)
        #     plot = plot/np.max(plot)*255
        #     image = Image.fromarray(plot).convert("L")
        #     image.save(os.path.join('.',f"{self.split}_{idx}_{i}.png"))
        
        return patch, label
        
class IRDatasetProcessor(VisionDataset):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.data_dir = params["train_input"]["data_dir"]

        self.num_classes = params["train_input"]["num_classes"]
        self.image_shape = params["train_input"]["image_shape"]  # of format (H, W, C)
        self.duplicate_act_worker_data = params["runconfig"].get(
            "duplicate_act_worker_data", False
        )

        self.shuffle_seed = params["train_input"].get("shuffle_seed", None)
        if self.shuffle_seed:
            torch.manual_seed(self.shuffle_seed)

        self.augment_data = params["train_input"].get("augment_data", True) 
        self.shuffle = params["train_input"].get("shuffle", True)

        # Multi-processing params.
        self.num_workers = params["train_input"].get("num_workers", 0)
        self.drop_last = params["train_input"].get("drop_last", True)
        self.prefetch_factor = params["train_input"].get("prefetch_factor", 10)
        self.input_dim = params["train_input"]["input_dim"]
        self.persistent_workers = params.get("persistent_workers", True)
        self.world_size = params["runconfig"]["world_size"]
        
        self.mp_type = torch.float32
        
        # default is that each activation worker sends `num_workers`
        # batches so total batch_size * num_act_workers * num_pytorch_workers samples

        # Using Faster Dataloader for mapstyle dataset.

            
    def create_dataset(self, is_training):
        split = "train" if is_training else "val"
        dataset = IRDataset(
            root=self.data_dir,
            split=split,
            input_dim=self.input_dim,
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
            do_horizontal_flip = torch.rand(size=(1,)).item() > 0.5
            # n_rots in range [0, 3)
            n_rotations = torch.randint(low=0, high=3, size=(1,)).item()

            if self.image_shape[0] != self.image_shape[1]:  # H != W
                # For a rectangle image
                n_rotations = n_rotations * 2
            h = torch.randint(high=image.shape[1]-self.image_shape[0]-1,size=(1,))
            w = torch.randint(high=image.shape[2]-self.image_shape[1]-1,size=(1,))
            delta = torch.rand(1) * 0.5
            augment_transform_image = self.get_augment_transforms(
                do_horizontal_flip=do_horizontal_flip,
                n_rotations=n_rotations,
                do_random_brightness=True,
                delta = delta,
                crop_h=h,
                crop_w=w,
                image_height=self.image_shape[0],
                image_width=self.image_shape[1]
            )
            augment_transform_mask = self.get_augment_transforms(
                do_horizontal_flip=do_horizontal_flip,
                n_rotations=n_rotations,
                do_random_brightness=False,
                delta = delta,
                crop_h=h,
                crop_w=w,
                image_height=self.image_shape[0],
                image_width=self.image_shape[1]
            )

            image = augment_transform_image(image)
            mask = augment_transform_mask(mask)

        # Handle dtypes and mask shapes based on `loss_type`
        # and `mixed_precsion`


        image = image.type(self.mp_type)
        mask = mask.type(torch.int64)
        return image, mask

    def get_augment_transforms(
        self, do_horizontal_flip, n_rotations, do_random_brightness, delta, crop_h, crop_w, image_height, image_width
    ):
        augment_transforms_list = []
        if self.is_training:
            if image_height is not None:
                crop_transform = transforms.Lambda(
                    lambda x: transforms.functional.crop(x, top=crop_h, left=crop_w, height=image_height, width=image_width)
                )
                augment_transforms_list.append(crop_transform)
            
            if do_horizontal_flip:
                horizontal_flip_transform = transforms.Lambda(
                    lambda x: transforms.functional.hflip(x)
                )
                augment_transforms_list.append(horizontal_flip_transform)

            if n_rotations > 0:
                rotation_transform = transforms.Lambda(
                    lambda x: rotation_90_transform(x, num_rotations=n_rotations)
                )
                augment_transforms_list.append(rotation_transform)
                
            if do_random_brightness:
                brightness_transform = transforms.Lambda(
                    lambda x: adjust_brightness_transform(x, p=0.5, delta=delta)
                )
                augment_transforms_list.append(brightness_transform)
            
        return transforms.Compose(augment_transforms_list)
