from __future__ import print_function  # print('me') instead of print 'me'
from __future__ import division  # 1/2 == 0.5, not 0
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
from PIL import Image # No need for ImageChops
import math
from skimage import img_as_float
from skimage.measure import compare_mse as mse


import nibabel as nib
def translation_transform(npArray, x, y, z):
#     assert: x<npArray.shape[0];
#     &&ynpArray.shape[1]&&z<npArray.shape[2]
    
    result_ndArray = np.zeros(npArray.shape)
    t1_transform_moved[x:, y:, z:] = t1_transform[:-x, :-y, :-z]
    return result_ndArray


t1_img = nib.load('data/a01.nii.gz')
t1_data = t1_img.get_data()

t1_transform= np.zeros(t1_data.shape)
t1_transform[:, :, :]= t1_data[:, :, :]


t1_transform_moved = np.zeros(t1_data.shape)
t1_transform_moved[15:, :, :] = t1_transform[:-15, :, :]



affine = np.diag([1, 2, 3, 1])
array_img = nib.Nifti1Image(t1_transform_moved, affine)


array_img.to_filename('my_image_again.nii')

# 
# plt.imshow(np.hstack((t1_slice, t2_slice_moved)))
# plt.show()
