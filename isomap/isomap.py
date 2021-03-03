#!/usr/bin/python3


## Import libraries
from time import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import offsetbox
from sklearn import manifold, datasets


## Constants
PLOT_ANNOT = 0


## Functions

## Plot single black and white images
def plt_image(img_array):
    img_rows=8
    img_cols=8
    img=np.empty((img_rows, img_cols, 3))
    for i in range(img_rows):
        for j in range(img_cols):
            img[i][j][0]=1-img_array[i*img_rows+j]/16.
            img[i][j][1]=1-img_array[i*img_rows+j]/16.
            img[i][j][2]=1-img_array[i*img_rows+j]/16.
    return(img)


# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    # Standardize the data for plotting
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    # Try to avoid the figure / axes to be turned without any reason
    # Try to have the 4 and 0 clusters always on the top right
    X_0 = X[y==0]
    X_4 = X[y==4]
    subMeanX = 0.5*(np.mean(X_4[:,0])+np.mean(X_0[:,0]))
    subMeanY = 0.25*(3*np.mean(X_4[:,1])+np.mean(X_0[:,1]))
    if (subMeanX<0.5): X[:,0] = 1-X[:,0]
    if (subMeanY<0.5): X[:,1] = 1-X[:,1]

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if (PLOT_ANNOT):
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)

    #plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


################
##    MAIN
################

## Load data
digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

## Look at data
print("Data size: ", X.shape)
print("Image array: ", X[0])
plt.figure()
plt.title("A digit image")
plt.imshow(plt_image(X[0]))


## Isomap projection of the digits dataset
for n_neighbors in range(5,41,5):
    # Compute isomap
    print("Computing Isomap projection for n_neighbors = %d" %n_neighbors)
    t0 = time()
    X_iso = manifold.Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(X)
    t1 = time()
    print("Done.")
    plot_embedding(X_iso,
               "Isomap projection of the digits (n_neighbors=%d)\n(time %.2fs)" %
               (n_neighbors ,t1-t0))
    #plt.savefig("fig/"+str(n_neighbors)+".png")


#plt.close('all')
plt.show()
