import os
import h5py
import numpy as np
from PIL import Image


def normolize(IR):
    negative_pos = np.where(IR<0)
    IR[negative_pos] = 0
    IR = IR/np.max(IR)
    return IR

def data_split( IR, label, train_test_split):
    split_col = int(IR.shape[1]*train_test_split)
    train_IR = IR[:,0:split_col]
    test_IR = IR[:,split_col:]
    train_label = label[:,:split_col]
    test_label = label[:,split_col:]
    
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
        
    img.save("label3.jpg")
    # IR = self.IR_patches[i][4] + (0.0001**0.5)*torch.randn(self.IR_patches[i][4].shape)
    # img = Image.fromarray(np.array((IR-torch.min(IR))/torch.max(IR)*255,dtype=np.uint8))
    # img.save(f"{self.mode}_IR_{i}.jpg")
            
    return train_IR, test_IR, train_label, test_label

if __name__ == "__main__":
    data_dir = './'
    IR = np.array(h5py.File(os.path.join(data_dir,'IR.mat'), 'r')['X'])
    IR = normolize(IR)
    label = np.array(h5py.File(os.path.join(data_dir,'Class.mat'), 'r')['CL'])
    IR = np.moveaxis(IR, 0, -1)

    train_IR, test_IR, train_label, test_label = data_split(IR, label, 0.79)
    with open(os.path.join(data_dir,'train_IR'),'wb') as f:
        np.save(f, train_IR)
    with open(os.path.join(data_dir,'test_IR'),'wb') as f:
        np.save(f, test_IR)
    with open(os.path.join(data_dir,'train_label'),'wb') as f:
        np.save(f, train_label)
    with open(os.path.join(data_dir,'test_label'),'wb') as f:
        np.save(f, test_label)
