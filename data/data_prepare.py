import os
import h5py
import re
import scipy.io
import shutil
import numpy as np
from glob import glob
from PIL import Image as im
from tqdm import tqdm

def normolize(IR, max_val):
    negative_pos = np.where(IR<0)
    IR[negative_pos] = 0
    IR = IR/max_val
    return IR


        
def save_data(IR_files, folder, max_IR, plot=True):
    for i, IR_file in enumerate(tqdm(IR_files)):
        idx = IR_file.split('/')[-1].split('.')[0]
        data = scipy.io.loadmat(IR_file)
        IR = data['IRsub']
        label = data['6S_weakLabel']
        IR = np.moveaxis(IR,2,0)
        IR = normolize(IR, max_IR)
        data_save_plot(folder,
                       IR, 
                       label, 
                       f'IR_{i}_{idx}',
                       f'label_{i}_{idx}',
                       plot)
               
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
    Label_patch = (Label_patch/6*255).astype(np.uint8)
    
    img = im.fromarray(IR_patch)
    img.save(os.path.join(save_folder, 'IR', f'{IR_save_name}.jpeg'))
    img = im.fromarray(Label_patch)
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
    
    save_data(patches_train, train_folder, max_IR, plot)
    save_data(patches_test, test_folder, max_IR, plot)

if __name__ == "__main__":
    train_folder = '/raid/projects/hangzheng/data/train'
    test_folder = '/raid/projects/hangzheng/data/val'
    plot = True
    IR_files = sorted(glob("/raid/projects/hangzheng/BR1003_Cores/Data/*"))
    
    train_test_split = 0.8
    create_folder(train_folder)
    create_folder(test_folder)

    prepare_data(IR_files,
                       train_test_split,
                       train_folder,
                       test_folder,
                       plot)
