import os
import h5py
import numpy as np
from PIL import Image
import shutil
def normolize(IR):
    negative_pos = np.where(IR<0)
    IR[negative_pos] = 0
    IR = IR/np.max(IR)
    return IR

def data_plot(label):
    img = Image.fromarray(np.array(label/6*255,dtype=np.uint8))
    
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
        img = Image.fromarray(np.array(label,dtype=np.uint8))
        img.save(f"label{i+1}.jpg")  
    return

def save_data(IR, label, patches, img_size, folder):
    for i, pt in enumerate(patches):
        IR_patch = IR[pt[1]:pt[1]+img_size,pt[0]:pt[0]+img_size,:]
        Label_patch = label[pt[1]:pt[1]+img_size,pt[0]:pt[0]+img_size]
        
        with open(os.path.join(folder, 'IR', f'IR_{i}'), 'wb') as f1, \
            open(os.path.join(folder, 'label', f'label_{i}'), 'wb') as f2:
            np.save(f1, IR_patch)
            np.save(f2, Label_patch)
        # img = Image.fromarray(np.array(IR_patch[:,:,0]/np.max(IR_patch[:,:,0])*255, dtype=np.uint8))
        # img.save(os.path.join(folder, 'IR', f'IR_{i}.jpeg'))
        # img = Image.fromarray(np.array(Label_patch/7*255, dtype=np.uint8))
        # img.save(os.path.join(folder, 'label', f'label_{i}.jpeg'))
        
def create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    os.mkdir(os.path.join(path,'IR'))
    os.mkdir(os.path.join(path,'label'))
            
if __name__ == "__main__":
    data_dir = './'
    train_folder = './train'
    test_folder = './test'
    create_folder(train_folder)
    create_folder(test_folder)
        
        
    IR = np.array(h5py.File(os.path.join(data_dir,'IR.mat'), 'r')['X'])
    IR = normolize(IR)  
    label = np.array(h5py.File(os.path.join(data_dir,'Class.mat'), 'r')['CL'])  # [H, W]
    IR = np.moveaxis(IR, 0, -1) # [H, W, C]
    img_size = 230
    train_test_split = 0.8
    with open("cell_centers.txt", "r") as f:
        pts = f.readlines()
        pts = [pt.split() for pt in pts]
        patches = []
        for pt in pts:
            for i in range(4):
                patches.append([int(float(pt[0])) - img_size*(i%2), int(float(pt[1])) - img_size*(i//2)])
        np.random.shuffle(patches)
        patches_train = patches[:int(len(patches)*train_test_split)]
        patches_test = patches[int(len(patches)*train_test_split):]
        
        save_data(IR,label, patches_train, img_size, train_folder)
        save_data(IR,label, patches_test, img_size, test_folder)
    
    data_plot(label)
    
