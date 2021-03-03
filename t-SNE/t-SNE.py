#!/usr/bin/python3


############################## READ ME ##############################
#
# DESCRIPTION
# * 2D visualisation using t-SNE of a 64-dimensional dataset
#   composed of 8x8 pixels images representing digits from 0 to 9
# 
# RESULTS
# * t-SNE reveals clear clusters. Some digits are divided in
#   subclusters (probably due to variation in the drawing of the
#   digit)
# * Running random init or PCA init does not affect much the
#   2D visualisation
# * When the perplexity is low, the clusters have a clear internal
#   structure (points are located on curved line). This internal
#   structure seems to vanish (at least it descreases) when the
#   perplexity increases.
# * The learning rate has no clear effect on this dataset for
#   values between 10 and 1000
# * The early_exaggeration has no clear effect on this dataset
#   for values between 4 and 20
#
#####################################################################


## Import libraries
from time import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import offsetbox
from sklearn import manifold, datasets


## Constants
RANDOM_STATE = np.random.randint(10000)
print("RANDOM_STATE = %d" %RANDOM_STATE)

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
digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

## Look at data
print("Data size: ", X.shape)
print("Image array: ", X[0])
plt.figure()
plt.title("A digit image")
plt.imshow(plt_image(X[0]))


## t-SNE embedding of the digits dataset
for ee in [12]:
    for l in [100]:
        for p in [30]: #[5,10,20,30,40,50]:
            print("Computing t-SNE embedding - init PCA - p=%d - l=%d - ee=%d" %(p,l,ee))
            tsne = manifold.TSNE(n_components=2, init='pca', perplexity=p,
                             learning_rate=l, early_exaggeration=ee,
                             random_state=RANDOM_STATE)
            t0 = time()
            X_tsne = tsne.fit_transform(X)
            t1 = time()
            plot_embedding(X_tsne, "t-SNE embedding - init PCA - perplexity=%d learning_rate=%d\nearly-exaggeration=%d - (time %.2fs)" % (p,l,ee,t1-t0))


## t-SNE embedding of the digits dataset
'''
print("Computing t-SNE embedding - init random")
tsne = manifold.TSNE(n_components=2, init='random', random_state=RANDOM_STATE)
t0 = time()
X_tsne = tsne.fit_transform(X)
t1 = time()
plot_embedding(X_tsne, "t-SNE embedding - init random\n(time %.2fs)" % (t1-t0))
'''

plt.show()
