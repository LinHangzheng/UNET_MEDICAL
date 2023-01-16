import os
import h5py
import numpy as np



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
