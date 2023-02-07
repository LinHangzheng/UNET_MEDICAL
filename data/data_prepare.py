import os
import h5py
import re
import scipy.io
import shutil
import numpy as np
from glob import glob
from PIL import Image as im

def normolize(IR):
    negative_pos = np.where(IR<0)
    IR[negative_pos] = 0
    IR = IR/np.max(IR)
    return IR


def small_data_plot_all_channel(label):
    label = (label/6*255).astype(np.uint8)
    img = im.fromarray(label)
    
    img.save("label.jpg")
    pos = { 0 : np.where(label==0),
            1 : np.where(label==1),
            2 : np.where(label==2),
            3 : np.where(label==3),
            4 : np.where(label==4),
            5 : np.where(label==5),
            6 : np.where(label==6)}
    for i in range(7):
        label = label*0
        label[pos[i]] = 255
        img = im.fromarray(label)
        img.save(f"label{i+1}.jpg")  
    return


def save_small_data(IR, label, patches, img_size, folder, plot=True):
    for i, pt in enumerate(patches):
        IR_patch = IR[:,pt[1]-img_size//2:pt[1]+img_size//2,pt[0]-img_size//2:pt[0]+img_size//2]
        Label_patch = label[pt[1]-img_size//2:pt[1]+img_size//2,pt[0]-img_size//2:pt[0]+img_size//2]
        
        data_save_plot(folder,
                       IR_patch, 
                       Label_patch, 
                       f'IR_{i}',
                       f'label_{i}',
                       plot)
        
def save_large_data(IR_files, folder, plot=True):
    for i, IR_file in enumerate(IR_files):
        folder_name = IR_file.split('/')[-2]
        idx = IR_file.split('/')[-1].split('.')[0]
        IR = scipy.io.loadmat(IR_file)['IR']
        label = scipy.io.loadmat(re.sub('IR','Class',IR_file))['CL']
        IR = np.moveaxis(IR,2,0)
        IR = normolize(IR)
        data_save_plot(folder,
                       IR, 
                       label, 
                       f'IR_{i}_{folder_name}_{idx}',
                       f'label_{i}_{folder_name}_{idx}',
                       plot)
               
def data_save_plot(save_folder,
                   IR_patch, 
                   Label_patch, 
                   IR_save_name, 
                   Label_save_name, 
                   plot=True):
    # save data into npy files
    np.save(os.path.join(save_folder, 'IR', IR_save_name),np.array(IR_patch))
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


def prepare_small_data(data_dir,
                       cell_centers_file, 
                       img_size, 
                       train_test_split,
                       train_folder,
                       test_folder,
                       plot=True):
    IR = np.array(h5py.File(os.path.join(data_dir,'IR.mat'), 'r')['X'])
    IR = normolize(IR)  
    label = np.array(h5py.File(os.path.join(data_dir,'Class.mat'), 'r')['CL'])  # [H, W]
    
    with open(cell_centers_file, "r") as f:
        pts = f.readlines()
        patches = [[float(p) for p in pt.split()] for pt in pts]
        patches = np.array(patches,dtype=np.int32)
        np.random.shuffle(patches)
        patches_train = patches[:int(len(patches)*train_test_split)]
        patches_test = patches[int(len(patches)*train_test_split):]
        
        save_small_data(IR, label, patches_train, img_size, train_folder, plot)
        save_small_data(IR, label, patches_test, img_size, test_folder, plot)
    return label


def prepare_large_data(large_IR_files,
                       train_test_split,
                       train_folder,
                       test_folder,
                       plot=True):

    np.random.shuffle(large_IR_files)
    patches_train = large_IR_files[:int(len(large_IR_files)*train_test_split)]
    patches_test = large_IR_files[int(len(large_IR_files)*train_test_split):]
    
    save_large_data(patches_train, train_folder, plot)
    save_large_data(patches_test, test_folder, plot)

if __name__ == "__main__":
    data_dir = './'
    train_folder = './train'
    test_folder = './val'
    plot = True
    large_IR_files = sorted(glob("/home/hangzheng/tissue_segmentation/data/IR/*/*"))
    
    small_img_size = 500
    train_test_split = 0.8
    create_folder(train_folder)
    create_folder(test_folder)

    # # prepare the low resolution image and plot their labels
    # label = prepare_small_data(data_dir, 
    #                            "cell_centers.txt", 
    #                            small_img_size, 
    #                            train_test_split, 
    #                            train_folder, 
    #                            test_folder,
    #                            plot)
    # small_data_plot_all_channel(label)

    prepare_large_data(large_IR_files,
                       train_test_split,
                       train_folder,
                       test_folder,
                       plot)

