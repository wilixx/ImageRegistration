'''
Created on Jan 30, 2018

@author: Dr.Guo
'''
import nibabel as nib
import matplotlib.pyplot as plt
import pylab
# img = nib.load('data/someones_anatomy.nii.gz')

img = nib.load('data/training_axial_crop_pat0.nii.gz')
img_2 = nib.load('data/training_axial_crop_pat0-label.nii.gz')
 # The 3x3 part of the affine is diagonal with all +ve values
img.affine
img_2.affine
# array([[  2.75,   0.  ,   0.  , -78.  ],
#        [  0.  ,   2.75,   0.  , -91.  ],
#        [  0.  ,   0.  ,   2.75, -91.  ],
#        [  0.  ,   0.  ,   0.  ,   1.  ]])
img_data = img.get_data()
img_2_data = img_2.get_data()
print(img_data.shape)
print(img_2_data.shape)

# a_slice = img_data[:, :, 28]
a_slice = img_data[:, :, 56]
# a_slice_2 = img_2_data[:, :, 28]
a_slice_2 = img_2_data[:, :, 56]
print("Let s go now")
# Need transpose to put first axis left-right, second bottom-top
# plt.imshow(a_slice)
plt.imshow(a_slice.T, cmap="gray", origin="lower")
# plt.imshow(a_slice_2.T, cmap="gray", origihttps://pypi.python.org/pypi/MedPyn="lower")
# plt.imshow(a_slice.T, cmap="Blues", origin="lower")
# plt.imshow(a_slice.T, cmap="Blues", origin="lower")
pylab.show()
# a_slice.show()
print("That is the end")
# plt.imshow("data/image.png")