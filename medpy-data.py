from medpy.io import load
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

i, h = load("data/someones_anatomy.nii.gz")

i[np.random.randint(0, i.shape[0], int(0.05 * i.size)), np.random.randint(0, i.shape[1], int(0.05 * i.size))] = i.min()
i[np.random.randint(0, i.shape[0], int(0.05 * i.size)), np.random.randint(0, i.shape[1], int(0.05 * i.size))] = i.max()

# plt.imshow(i, cmap = cm.Greys_r)
plt.imshow(i)