#!/usr/bin/python3


########################## READ ME ##########################
#
# DESCRIPTION
# === PCA ===
# * Principal Component Analysis (PCA) of a set of features
#   describing how athletes perform in decathlon
# * Plot of rank in competitions in the PC basis
# * Plot of features in the PC basis
# * Matrix of features in the PC basis
#             PC in the features basis
# * Plot of angles between PC vectors
# 
# === FA ===
# * Factor Analysis (FA) of a set of features
#   describing how athletes perform in decathlon
# * Plot of rank in competitions in the FA basis
# * Matrix of FA components in the features basis
# * Plot of angles between FA vectors
#
#
# RESULTS
# === PCA ===
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
# === FA ===
# * Plot of rank in PC basis and FA basis are vry similar
# * Matrix of FA vectors in the features basis shows that
#   some FA vectors are null: this features of the FA
#   algorithm was expected
# * Matrix of FA vectors in the features basis shows that
#   FA vectors are not orthogonal: this was expected too.
# * The coefficients of the decomposition of PC and non null
#   FA vectors in the features basis are quite close. The
#   PCA and FA do not differ much in this case.
#
#
#############################################################



######################
##  Load librairies
######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import offsetbox
from sklearn import preprocessing
from sklearn import decomposition, datasets


################
##  CONSTANTS
################
PCA = 0
FA  = 0
NMF = 1


PLOT_ANNOT = 0

################
##  Functions
################

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


## Scale and visualize the embedding vectors
def plot_embedding(X, title=None, comp=[0,1]):
    # Standardize the data for plotting
    #x_min, x_max = np.min(X, 0), np.max(X, 0)
    #X = (X - x_min) / (x_max - x_min)
    for i in range(n_features):
        feat_min = min(X[:,i])
        feat_max = max(X[:,i])
        if (feat_min!=feat_max): X[:,i] = (X[:,i]-feat_min)/(feat_max-feat_min)
        else: X[:,i] = X[:,i] - feat_min

    plt.figure()
    ax = plt.subplot(111)
    plt.xlabel("Component "+str(comp[0]))
    plt.ylabel("Component "+str(comp[1]))
    #plt.xlim([10**-2,1])
    #plt.ylim([10**-2,1])
    #plt.xscale('log')
    #plt.yscale('log')
    for i in range(X.shape[0]):
        plt.text(X[i, comp[0]], X[i, comp[1]], str(y[i]),
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


##################
##     MAIN
##################

NFIG = 0

## Load data
digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
print("Data size: ", X.shape)

## Preprocessing
X_scaled = preprocessing.StandardScaler().fit_transform(X)      # scale features so that they have an
                                                                # average 0 and std dev 1: f <- (f-f*)/s(f) 
'''
X_scaled_NMF = np.zeros_like(X, dtype=float)
for i in range(n_features):
    feat_min = min(X[:,i])
    feat_max = max(X[:,i])
    X_scaled_NMF[:,i] = (X[:,i]-feat_min)/(feat_max-feat_min)
'''


#  ##############################################################################
## Principal Component Analysis

if (PCA):
    print("\n===== Principal Component Analysis =====")
    pca = decomposition.PCA(n_components=n_features)
    print("Fit...")
    pca.fit(X)

    explained_variance_ratio = pca.explained_variance_ratio_
    print("Principal component\t% of variance explained\tCumulative sum")
    for i,(r,rc) in enumerate(zip(explained_variance_ratio, explained_variance_ratio.cumsum())):
        print("%d\t\t\t%.3f\t\t\t%.3f" %(i+1,r,rc))
    #print("Overall variance explained: %.3f" %pca.explained_variance_ratio_.sum())


    # Plot observations in the PC basis and label them using rank feature
    X_projected = pca.transform(X)   # Project X on principal components
    plot_embedding(X_projected, "PCA")
    NFIG+=1

    # Plot the features in the 2 main principal components basis
    pc_featBasis = pca.components_              # Principal axes in feature space
    feat_pcBasis = np.transpose(pc_featBasis)   # Features in principal axes space

    '''
    NFIG+=1
    plt.figure(NFIG)
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
    '''

    # Features in Principal Component basis
    NFIG+=1
    plt.figure(NFIG)
    plt.title("Features in the Principal Component basis")
    axe = plt.gca()
    axe.invert_yaxis()
    axe.xaxis.tick_top()
    sns.heatmap(np.transpose(feat_pcBasis), annot=False, cbar=True)


    # PC in features basis
    NFIG+=1
    plt.figure(NFIG)
    plt.title("Principal Components in the features basis")
    axe = plt.gca()
    axe.invert_yaxis()
    axe.xaxis.tick_top()
    sns.heatmap(np.transpose(pc_featBasis), annot=False, cbar=True)


    # Angle between PC basis vectors
    eps = np.array([10**-5 for i in range(n_features)])
    angles = np.full_like(pc_featBasis, 0, dtype=int)
    for i in range(n_features):
        for j in range(n_features):
            if ( (abs(pc_featBasis[i,:])>=eps).any() and (abs(pc_featBasis[j,:])>=eps).any() ):
                norm1=np.sqrt(np.dot(pc_featBasis[i,:],pc_featBasis[i,:]))
                norm2=np.sqrt(np.dot(pc_featBasis[j,:],pc_featBasis[j,:]))
                scal_prod = np.dot(pc_featBasis[i,:],pc_featBasis[j,:])/(norm1*norm2)
                if (scal_prod<-1): scal_prod=-1
                if (scal_prod>1): scal_prod=1
                angles[i,j]=round(np.arccos(scal_prod)*(180./np.pi))

    NFIG+=1
    plt.figure(NFIG)
    plt.title("Angle between PC basis vectors")
    axe = plt.gca()
    axe.invert_yaxis()
    axe.xaxis.tick_top()
    sns.heatmap(angles, annot=False, cbar=True)


    # Explained variance as a function as the number of PC
    NFIG+=1
    plt.figure(NFIG)
    plt.title("Cumulative explained variance as a function\nof the number of Principal Components")
    plt.plot( [i for i in range(0,n_features+1)], [0]+list(explained_variance_ratio.cumsum()),
              color="steelblue",
              linestyle="-" )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative explained variance")
    plt.xlim([0,n_features])
    plt.ylim([0,1])


    # Explained variance for each PC
    NFIG+=1
    plt.figure(NFIG)
    plt.title("Explained variance for each Principal Component")
    plt.plot( [i for i in range(1,n_features+1)], list(explained_variance_ratio),
              color="steelblue",
              linestyle="-" )
    plt.xlabel("Principal Component Number")
    plt.ylabel("Explained variance")
    plt.xlim([0,n_features])
    plt.ylim([0,0.35])



#  ##############################################################################
## Factor Analysis

if (FA):
    print("\n===== Factor Analysis =====")
    fa = decomposition.FactorAnalysis(n_components=n_features)
    print("Fit...")
    fa.fit(X)

    # Plot observations in the FA basis and label them using rank feature
    X_faprojected = fa.transform(X)   # Project X on principal components
    plot_embedding(X_faprojected, "Factor Analysis")
    NFIG+=1


    # Plot the features in the 2 main principal components basis
    fa_featBasis = fa.components_                # Principal axes in feature space


    # FA in features basis
    NFIG+=1
    plt.figure(NFIG)
    plt.title("FA components in the features basis")
    axe = plt.gca()
    axe.invert_yaxis()
    axe.xaxis.tick_top()
    sns.heatmap(np.transpose(fa_featBasis), annot=False, cbar=True)


    # Angle between FA basis vectors
    eps = np.array([10**-5 for i in range(n_features)])
    angles = np.full_like(fa_featBasis, 0, dtype=int)
    for i in range(n_features):
        for j in range(n_features):
            if ( (abs(fa_featBasis[i,:])>=eps).any() and (abs(fa_featBasis[j,:])>=eps).any() ):
                norm1=np.sqrt(np.dot(fa_featBasis[i,:],fa_featBasis[i,:]))
                norm2=np.sqrt(np.dot(fa_featBasis[j,:],fa_featBasis[j,:]))
                scal_prod = np.dot(fa_featBasis[i,:],fa_featBasis[j,:])/(norm1*norm2)
                if (scal_prod<-1): scal_prod=-1
                if (scal_prod>1): scal_prod=1
                angles[i,j]=round(np.arccos(scal_prod)*(180./np.pi))

    NFIG+=1
    plt.figure(NFIG)
    plt.title("Angle between FA basis vectors")
    axe = plt.gca()
    axe.invert_yaxis()
    axe.xaxis.tick_top()
    sns.heatmap(angles, annot=False, cbar=True)



#  ##############################################################################
## Non-Negative Matrix Factorisation (NMF)

if (NMF):
    print("\n===== Non-Negative Matrix Factorisation (NMF) =====")
    #  cd  300  0.
    solver='cd'
    alpha=240.0
    l1_ratio=1.0

    nmf = decomposition.NMF(n_components=n_features, solver=solver, beta_loss='frobenius',
                            tol=0.0001, random_state=None, alpha=alpha,
                            l1_ratio=l1_ratio, max_iter=1000)
    # alpha : coef of the penalisation term
    # l1_ratio : penalisation term = alpha * ( l1_ratio*| ? | + (1-l1_ratio)*|| ? || )

    print("Fit...")
    nmf.fit(X)
    print("Done.")


    # Plot the features in the 2 main principal components basis
    nmf_featBasis = nmf.components_                # Principal axes in feature space
    
    
    # Plot observations in the NMF basis and label them using rank feature
    X_nmfprojected = nmf.transform(X)   # Project X on principal components
    eps = np.array([10**-5 for i in range(n_features)])
    nonZeroComp = []
    comp = 0
    while (len(nonZeroComp)<2):
        if ( (abs(nmf_featBasis[comp,:])>=eps).any() ):
            nonZeroComp.append(comp)
        comp+=1
    plot_embedding(X_nmfprojected, "NMF", comp=nonZeroComp)
    NFIG+=1


    # NMF components in features basis
    NFIG+=1
    plt.figure(NFIG)
    plt.title("NMF components in the features basis\nsolver=%s alpha=%s l1_ratio=%s" %(solver, alpha, l1_ratio))
    axe = plt.gca()
    axe.invert_yaxis()
    axe.xaxis.tick_top()
    sns.heatmap(np.transpose(nmf_featBasis), annot=False, cbar=True)


    # Angle between NMF basis vectors
    eps = np.array([10**-5 for i in range(n_features)])
    angles = np.full_like(nmf_featBasis, 0, dtype=int)
    for i in range(n_features):
        for j in range(n_features):
            if ( (abs(nmf_featBasis[i,:])>=eps).any() and (abs(nmf_featBasis[j,:])>=eps).any() ):
                norm1=np.sqrt(np.dot(nmf_featBasis[i,:],nmf_featBasis[i,:]))
                norm2=np.sqrt(np.dot(nmf_featBasis[j,:],nmf_featBasis[j,:]))
                scal_prod = np.dot(nmf_featBasis[i,:],nmf_featBasis[j,:])/(norm1*norm2)
                if (scal_prod<-1): scal_prod=-1
                if (scal_prod>1): scal_prod=1
                angles[i,j]=round(np.arccos(scal_prod)*(180./np.pi))

    NFIG+=1
    plt.figure(NFIG)
    plt.title("Angle between NMF basis vectors\nsolver=%s alpha=%s l1_ratio=%s" %(solver, alpha, l1_ratio))
    axe = plt.gca()
    axe.invert_yaxis()
    axe.xaxis.tick_top()
    sns.heatmap(angles, annot=False, cbar=True)


#plt.close('all')
plt.show()

