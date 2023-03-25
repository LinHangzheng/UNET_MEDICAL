import os
import h5py
import re
import scipy.io
import shutil
import numpy as np
from glob import glob
from PIL import Image as im
from tqdm import tqdm

RGB_PALLET = np.array([
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
])

def normolize(IR, max_val):
    negative_pos = np.where(IR<0)
    IR[negative_pos] = 0
    IR = IR/max_val
    return IR



def save_val_patches(IR, label, img_size, folder, idx, name, plot):
    h_start = (IR.shape[1]-img_size*4)//2
    w_start = (IR.shape[2]-img_size*4)//2
    h_idx  = 0
    while h_idx < 4:
        w_idx = 0
        while w_idx < 4:
            h = h_start + h_idx*img_size
            w = w_start + w_idx*img_size
            IR_patch = IR[:,h:h+img_size,w:w+img_size]
            label_patch = label[h:h+img_size,w:w+img_size]
            
            data_save_plot(re.sub('val_large','val',folder),
                    IR_patch, 
                    label_patch, 
                    f'IR_{idx}_{name}_{h_idx}_{w_idx}',
                    f'label_{idx}_{name}_{h_idx}_{w_idx}',
                    plot)
            
            w_idx += 1
        h_idx += 1
            
def save_data(IR_files, folder, max_IR, plot=True, train=True):
    for i, IR_file in enumerate(tqdm(IR_files)):
        name = IR_file.split('/')[-1].split('.')[0]
        data = scipy.io.loadmat(IR_file)
        IR = data['IRsub']
        label = data['6S_weakLabel']
        IR = np.moveaxis(IR,2,0)
        IR = normolize(IR, max_IR)
        data_save_plot(folder,
                       IR, 
                       label, 
                       f'IR_{i}_{name}',
                       f'label_{i}_{name}',
                       plot)
        
        if not train:
            save_val_patches(IR, label, 224, folder, i, name, plot)
               
def data_save_plot(save_folder,
                   IR_patch, 
                   Label_patch, 
                   IR_save_name, 
                   Label_save_name, 
                   plot=True):
    # save data into npy files
    np.save(os.path.join(save_folder, 'IR', IR_save_name),np.array(IR_patch).astype(np.float32))
    np.save(os.path.join(save_folder, 'label', Label_save_name),np.array(Label_patch))
    
    # plot data
    if not plot:
        return
    
    IR_patch = (IR_patch[0,:,:]/np.max(IR_patch[0,:,:])*255).astype(np.uint8)
    labels_RGB = np.zeros([Label_patch.shape[0],Label_patch.shape[1],3]).astype(np.uint8)
    for k in range(7):
        labels_RGB[np.where(Label_patch==k)] = RGB_PALLET[k]
    img = im.fromarray(IR_patch)
    img.save(os.path.join(save_folder, 'IR', f'{IR_save_name}.jpeg'))
    img = im.fromarray(labels_RGB)
    img.save(os.path.join(save_folder, 'label', f'{Label_save_name}.jpeg'))
       
        
def create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(os.path.join(path,'IR'))
    os.mkdir(os.path.join(path,'label'))


def prepare_data(large_IR_files,
                       train_test_split,
                       train_folder,
                       test_folder,
                       plot=True):

    np.random.shuffle(large_IR_files)
    max_IR = 0
    for i, IR_file in enumerate(tqdm(large_IR_files)):
        IR = scipy.io.loadmat(IR_file)['IRsub']
        max_IR = max(max_IR, np.max(IR))
        
    patches_train = large_IR_files[:int(len(large_IR_files)*train_test_split)]
    patches_test = large_IR_files[int(len(large_IR_files)*train_test_split):]
    
    save_data(patches_train, train_folder, max_IR, plot, True)
    save_data(patches_test, test_folder, max_IR, plot, False)

if __name__ == "__main__":
    root = '/home/hangzheng/tissue_segmentation/data'
    train_folder = os.path.join(root, 'train')
    test_large_folder = os.path.join(root, 'val_large')
    test_folder = os.path.join(root, 'val')
    
    plot = True
    IR_files = sorted(glob("/home/hangzheng/tissue_segmentation/data/BR1003_Cores/Data/*"))
    
    train_test_split = 0.8
    create_folder(train_folder)
    create_folder(test_large_folder)
    create_folder(test_folder)

    prepare_data(IR_files,
                       train_test_split,
                       train_folder,
                       test_large_folder,
                       plot)
