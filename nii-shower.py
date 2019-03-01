'''
Created on Jan 30, 2018

@author: Dr.Guo
'''
import glob
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np


import nibabel as nib
import matplotlib.pyplot as plt
import pylab
import SimpleITK as sitk
import numpy as np

import visvis as vv
# from keras.backend.common import image_data_format
# img = nib.load('data/someones_anatomy.nii.gz')
def overlay(img1, img2, title=None, interpolation=None, sizeThreshold=128):
    """
    Description
    -----------
    Displays an overlay of two 2D images using matplotlib.pyplot.imshow().

    Args
    ----
    img1 : np.array or sitk.Image
        image to be displayed
    img2 : np.array or sitk.Image
        image to be displayed

    Optional
    --------
    title : string
        string used as image title
        (default None)
    interpolation : string
        desired option for interpolation among matplotlib.pyplot.imshow options
        (default nearest)
    sizeThreshold : integer
        size under which interpolation is automatically set to 'nearest' if all
        dimensions of img1 and img2 are below
        (default 128)
    """
    # Check for type of images and convert to np.array
    if isinstance(img1, sitk.Image):
        img1 = sitk.GetArrayFromImage(img1)
    if isinstance(img2, sitk.Image):
        img2 = sitk.GetArrayFromImage(img2)
    if type(img1) is not type(img2) is not np.ndarray:
        raise NotImplementedError('Please provide images as np.array or '
                                  'sitk.Image.')
    # Check for size of images
    if not img1.ndim == img2.ndim == 2:
        raise NotImplementedError('Only supports 2D images.')

    if interpolation:
        plt.imshow(img1, cmap='summer', interpolation=interpolation)
        plt.imshow(img2, cmap='autumn', alpha=0.5, interpolation=interpolation)
    elif max(max(img1.shape), max(img2.shape)) > sizeThreshold:
        plt.imshow(img1, cmap='summer')
        plt.imshow(img2, cmap='autumn', alpha=0.5)
    else:
        plt.imshow(img1, cmap='summer', interpolation='nearest')
        plt.imshow(img2, cmap='autumn', alpha=0.5, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def plot_3d(image, threshold=100):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p=p[::2,::2,::2]
#     verts, faces = measure.marching_cubes(p, threshold)
   # verts, faces,_,_ = measure.marching_cubes_lewiner(p, threshold)
    
    verts, faces,normals, values= measure.marching_cubes(p, threshold)
    '''
    verts, faces, normals, values = marching_cubes_lewiner(myvolume, 0.0) # doctest: +SKIP
      >>> vv.mesh(np.fliplr(verts), faces, normals, values) # doctest: +SKIP
      >>> vv.use().Run() # doctest: +SKIP
      
    '''
    
    
    lung_mesh=vv.mesh(np.fliplr(verts), faces, normals, values, 
            verticesPerFace=4,colormap=None, clim=None, texture=None, axesAdjust=True, axes=None)
#     lung_mesh.setcolor([0.45, 0.45, 0.75])
    faceColor = 'g'
    vv.use().Run()
    
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()
    

img = nib.load('data/training_axial_crop_pat0.nii.gz')
img = nib.load('data/a20-seg.nii/a20-seg.nii')
img = nib.load('data/result_a01-new.nii.gz')
img = nib.load('data/overlay_a01-new.nii.gz')
img = nib.load('data/a01-seg.nii.gz')
img = nib.load('data/a01.nii.gz')
img_2 = nib.load('data/training_axial_crop_pat0-label.nii.gz')
img_2 = nib.load('data/result_a01-new.nii.gz')
 # The 3x3 part of the affine is diagonal with all +ve values
img.affine
img_2.affine
# array([[  2.75,   0.  ,   0.  , -78.  ],
#        [  0.  ,   2.75,   0.  , -91.  ],
#        [  0.  ,   0.  ,   2.75, -91.  ],
#        [  0.  ,   0.  ,   0.  ,   1.  ]])
img_data = img.get_data()
img_2_data = img_2.get_data()

# overlay(img_data, img_2_data)


print(img_data.shape)
print(img_2_data.shape)

plot_3d(img_data)
print(img_2_data)

# a_slice = img_data[:, :, 28]
a_slice = img_data[:, :, 100]
print(a_slice.shape)
print(a_slice)
# a_slice.transpose(2,1,0)
# a_slice_2 = img_2_data[:, :, 28]
a_slice_2 = img_2_data[:, :, 100]
print("Let s go now")
overlay(a_slice.T, a_slice_2.T)
# Need transpose to put first axis left-right, second bottom-top
# plt.imshow(a_slice)
# plt.imshow(a_slice.T, cmap="gray", origin="lower")

# plt.imshow(a_slice.T, origin="lower")
plt.imshow(a_slice.T, origin="lower")
pylab.show()

plt.imshow(a_slice_2.T, origin="lower")
# plt.imshow(a_slice_2.T, cmap="gray", origihttps://pypi.python.org/pypi/MedPyn="lower")
# plt.imshow(a_slice.T, cmap="Blues", origin="lower")
# plt.imshow(a_slice.T, cmap="Blues", origin="lower")
pylab.show()
# a_slice.show()
print("That is the end")
# plt.imshow("data/image.png")