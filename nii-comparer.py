'''
Created on Jan 30, 2018

@author: Dr.Guo
'''
import nibabel as nib
import matplotlib.pyplot as plt
import pylab
# img = nib.load('data/someones_anatomy.nii.gz')

img_1 = nib.load('data/training_axial_crop_pat0.nii.gz')
img_2 = nib.load('data/training_axial_crop_pat0-label.nii.gz')
 # The 3x3 part of the affine is diagonal with all +ve values
img_1.affine
img_2.affine
# array([[  2.75,   0.  ,   0.  , -78.  ],
#        [  0.  ,   2.75,   0.  , -91.  ],
#        [  0.  ,   0.  ,   2.75, -91.  ],
#        [  0.  ,   0.  ,   0.  ,   1.  ]])
img_1_data = img_1.get_data()
img_2_data = img_2.get_data()
print(img_1_data.shape)
print(img_2_data.shape)

# a_slice = img_data[:, :, 28]
a_slice_1 = img_1_data[:, :, 56]
# a_slice_2 = img_2_data[:, :, 28]
a_slice_2 = img_2_data[:, :, 56]
a_=a_slice_1-a_slice_2
print("Let s go now")
print(img_2_data[25, 25, 56])

# Need transpose to put first axis left-right, second bottom-top
# plt.imshow(a_slice)
# plt.imshow(a_slice_1.T, cmap="gray", origin="lower")
# plt.imshow(a_slice_2.T, cmap="gray", origin="lower")
plt.imshow(a_.T, cmap="gray", origin="lower")
# plt.imshow(a_slice.T, cmap="Blues", origin="lower")
# plt.imshow(a_slice.T, cmap="Blues", origin="lower")
pylab.show()
print(img_1_data[32, 56, :])
# a_slice.show()
print("That is the end")
# plt.imshow("data/image.png")