#!/usr/bin/python3


########################## READ ME ##########################
#
# DESCRIPTION
# * Principal Component Analysis (PCA) of a set of features
#   describing how athletes perform in decathlon
# * Plot of rank in competitions in the PC basis
# * Plot of features in the PC basis
#
# RESULTS
# * High values of PC1 indicate good performance: sports
#   that require low socre (like time of a race) have a
#   a negative PC1 component while sport that require high
#   score (like the height of a jump) have a positive PC1
#   component.
# * This can be checked on the rank in the (PC1, PC2) basis:
#   low ranks (i.e. good performance) are obtained
#   for PC1>0.
# * Some variables are quite correlated, like discuss and
#   Shot put
#
#############################################################




## Load librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import decomposition


## Load data
data = pd.read_csv('./input/decathlon.txt', sep="\t")

## Preprocessing
data_pca = data.drop(['Points', 'Rank', 'Competition'], axis=1) # remove useless columns for PCA
X = data_pca.values                                             # transform data into a numpy array
X = preprocessing.StandardScaler().fit_transform(X)             # scale features so that they have an
                                                                # average 0 and std dev 1: f <- (f-f*)/s(f) 
print("Data size: ", X.shape)
n_features = X.shape[1]


## Principal Component analysis

print("\n===== Principal Component Analysis (PCA) =====")
pca = decomposition.PCA(n_components=n_features)  # start by looking at the first 2 PC
print("Fit...")
pca.fit(X)

explained_variance_ratio = pca.explained_variance_ratio_
print("Principal component\t% of variance explained\tCumulative sum")
for i,(r,rc) in enumerate(zip(explained_variance_ratio, explained_variance_ratio.cumsum())):
    print("%d\t\t\t%.3f\t\t\t%.3f" %(i+1,r,rc))
#print("Overall variance explained: %.3f" %pca.explained_variance_ratio_.sum())


# Plot observations in the PC basis and label them using rank feature
X_projected = pca.transform(X)   # Project X on principal components

plt.figure(1)
plt.title("Rank of athletes in the Principal Component basis")
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=data.get('Rank'))

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()


# Plot the features in the 2 main principal components basis
pc_featBasis = pca.components_              # Principal axes in feature space
feat_pcBasis = np.transpose(pc_featBasis)   # Features in principal axes space

plt.figure(2)
plt.title("Features in the Principal Component basis")
for i,feat in enumerate(feat_pcBasis):
    plt.plot([0, feat[0]], [0, feat[1]], color='k')  # Feature in the (PC1, PC2) basis
    plt.text(feat[0], feat[1], str(i)+' '+data.columns[i], fontsize='14')   # display features name

# Draw axis
plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')
plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')
plt.xlim([-0.7, 0.7])
plt.ylim([-0.7, 0.7])
plt.xlabel("PC1")
plt.ylabel("PC2")


# Features in Principal Component basis
plt.figure(3)
plt.title("Features in the Principal Component basis")
axe = plt.gca()
axe.invert_yaxis()
axe.xaxis.tick_top()
sns.heatmap(np.transpose(feat_pcBasis), annot=True, cbar=True)


# Explained variance as a function as the number of PC
plt.figure(4)
plt.title("Cumulative explained variance as a function\nof the number of Principal Components")
plt.plot( [i for i in range(0,n_features+1)], [0]+list(explained_variance_ratio.cumsum()),
          color="steelblue",
          linestyle="-" )
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative explained variance")
plt.xlim([0,n_features])
plt.ylim([0,1])


# Explained variance for each PC
plt.figure(5)
plt.title("Explained variance for each Principal Component")
plt.plot( [i for i in range(1,n_features+1)], list(explained_variance_ratio),
          color="steelblue",
          linestyle="-" )
plt.xlabel("Principal Component Number")
plt.ylabel("Explained variance")
plt.xlim([0,n_features])
plt.ylim([0,0.35])



## Kernel Principal Component analysis

print("\n===== Kernel Principal Component Analysis (kPCA) =====")
GAMMA=[ 0.01, 0.05, 0.1, 0.5, 1, 5, 10 ]

for i,g in enumerate(GAMMA):
    kpca = decomposition.KernelPCA(n_components=n_features, kernel="rbf", gamma=g)

    print("Fit...")
    kpca.fit(X)
 
    # Plot observations in the PC basis and label them using rank feature
    X_kprojected = kpca.transform(X)   # Project X on principal components
 
    plt.figure(6+i)
    plt.title("Rank of athletes in the kernel Principal Component basis\ngamma = "+str(g))
    plt.scatter(X_kprojected[:, 0], X_kprojected[:, 1], c=data.get('Rank'))
 
    #plt.xlim([-5, 5])
    #plt.ylim([-5, 5])
    plt.xlabel("kPC1")
    plt.ylabel("kPC2")
    plt.colorbar()



plt.close(2)
plt.close(3)
plt.close(4)
plt.close(5)
plt.show()
