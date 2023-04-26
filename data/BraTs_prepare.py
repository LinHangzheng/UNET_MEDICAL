import tarfile
import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
import PIL
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.util import montage 
# file = tarfile.open('brats-2021-task1/BraTS2021_Training_Data.tar')
# file.extractall('./brain_images')
# file.close()
TRAIN_DATASET_PATH = '/home/hangzheng/tissue_segmentation/data/brain_images/'
flair = '/home/hangzheng/tissue_segmentation/data/brain_images/BraTS2021_00002/BraTS2021_00002_flair.nii.gz'
seg = '/home/hangzheng/tissue_segmentation/data/brain_images/BraTS2021_00002/BraTS2021_00002_seg.nii.gz'
t1 = '/home/hangzheng/tissue_segmentation/data/brain_images/BraTS2021_00002/BraTS2021_00002_t1.nii.gz'
t1ce = '/home/hangzheng/tissue_segmentation/data/brain_images/BraTS2021_00002/BraTS2021_00002_t1ce.nii.gz'
t2 = '/home/hangzheng/tissue_segmentation/data/brain_images/BraTS2021_00002/BraTS2021_00002_t2.nii.gz'


flair = nib.load(flair).get_fdata()
seg = nib.load(seg).get_fdata()
t1 = nib.load(t1).get_fdata()
t1ce = nib.load(t1ce).get_fdata()
t2 = nib.load(t2).get_fdata()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 25
ax1.imshow(flair[:,:,flair.shape[0]//2-slice_w], cmap = 'gray')
ax1.set_title('Image flair')
ax2.imshow(t1[:,:,t1.shape[0]//2-slice_w], cmap = 'gray')
ax2.set_title('Image t1')
ax3.imshow(t1ce[:,:,t1ce.shape[0]//2-slice_w], cmap = 'gray')
ax3.set_title('Image t1ce')
ax4.imshow(t2[:,:,t2.shape[0]//2-slice_w], cmap = 'gray')
ax4.set_title('Image t2')
ax5.imshow(seg[:,:,seg.shape[0]//2-slice_w])
ax5.set_title('Mask')

fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(t1[50:-50,:,:]), 90, resize=True), cmap ='gray')

fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(seg[60:-60,:,:]), 90, resize=True), cmap ='gray')

niimg = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_flair.nii.gz')
nimask = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS2021_01261/BraTS2021_01261_seg.nii.gz')

fig, axes = plt.subplots(nrows=4, figsize=(30, 40))


nlplt.plot_anat(niimg,
                title='BraTS18_Training_001_flair.nii plot_anat',
                axes=axes[0])

nlplt.plot_epi(niimg,
               title='BraTS18_Training_001_flair.nii plot_epi',
               axes=axes[1])

nlplt.plot_img(niimg,
               title='BraTS18_Training_001_flair.nii plot_img',
               axes=axes[2])

nlplt.plot_roi(nimask, 
               title='BraTS18_Training_001_flair.nii with mask plot_roi',
               bg_img=niimg, 
               axes=axes[3], cmap='Paired')

plt.show()
pass