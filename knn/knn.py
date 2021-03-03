#!/usr/bin/python3

import numpy as np
import sys
import sklearn as sk
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


DATASET_SIZE = 5000
RANDOM_STATE = 3
TRAIN   = False
PREDICT = True
if (len(sys.argv)>=3):
    if (sys.argv[1]=='1'): TRAIN   = True
    if (sys.argv[2]=='0'): PREDICT = False


# ARGUMENTS WHEN RUNNING SCRIPT:
# ARG1: TRAIN    0: NO TRAINING    1: TRAINING
# ARG2: PREDICT  0: NO PREDICTION  1: PREDICTIONS


# Load data
mnist = fetch_mldata('MNIST original', data_home='input/')

# Le dataset principal qui contient toutes les images
print ("Size of features: {}".format(mnist.data.shape))

# Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
print ("Size of target: {}".format(mnist.target.shape))

# Sample
data, target = sk.utils.resample(mnist.data, mnist.target, n_samples=DATASET_SIZE, random_state=RANDOM_STATE)
#print(mnist.data.shape)
#print(mnist.target.shape)

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8, random_state=RANDOM_STATE)



if (TRAIN):

    k_list=[k for k in range(1,6,1)]
    score_list=np.zeros(len(k_list))
    for i,k in enumerate(k_list):
        print(k,"-NN:")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(xtrain, ytrain)
        score_list[i] = knn.score(xtest, ytest)
        print("   Score: {}".format(score_list[i]))

    plt.plot(k_list, score_list)
    plt.show()



if (PREDICT):
    # On récupère le classifieur le plus performant
    knn = KNeighborsClassifier(1)
    knn.fit(xtrain, ytrain)

    # On récupère les prédictions sur les données test
    predicted = knn.predict(xtest)

    # On redimensionne les données sous forme d'images
    images = xtest.reshape((-1, 28, 28))

    # On selectionne un echantillon de 12 images au hasard
    select = np.random.randint(images.shape[0], size=45)

    # On affiche les images avec la prédiction associée
    for index, value in enumerate(select):
        plt.subplot(5,9,index+1)
        plt.axis('off')
        plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
        plt.title('Predicted: %i' % predicted[value])

    plt.show()

