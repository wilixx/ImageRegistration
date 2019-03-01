import matplotlib.pyplot as plt
import nibabel as nib
import pylab
epi_img = nib.load('data/someones_epi.nii.gz')
epi_img_data = epi_img.get_data()
epi_img_data.shape
# (53, 61, 33)

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices)+1)
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[26, :, :]
slice_1 = epi_img_data[:, 30, :]
slice_2 = epi_img_data[:, :, 16]
slice_3 = epi_img_data[:, :, 16]
show_slices([slice_0, slice_1, slice_2, slice_3])
plt.suptitle("Center slices for EPI image")
pylab.show()