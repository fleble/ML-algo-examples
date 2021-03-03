#!/usr/bin/python3

import numpy as np
import sklearn as sk
import matplotlib.pylab as plt
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


DATASET_SIZE = 5000
RANDOM_STATE = 3


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


def prob_dict(preds):
    prob={i:0 for i in range(10)}
    n=len(preds)
    for i in range(10):
        for j in range(n):
            if (preds[j]==i): prob[i]+=1./float(n)
    return(prob)

def get_max_prob(prob_dict):
    return(max(prob_dict, key=prob_dict.get))

def get_k_max_prob(prob_dict,k):
    return(sorted(prob_dict.items(), key=lambda kv: -kv[1])[:k])


## Training single kNN
k_list = [k for k in range(1,12,2)]
n_knn = len(k_list)
acc = np.zeros(n_knn)
KNN = []

for i,k in enumerate(k_list):
    KNN.append(KNeighborsClassifier(n_neighbors=k))
    KNN[i].fit(xtrain, ytrain)
    acc[i] = np.sum(KNN[i].predict(xtest)==ytest)/len(ytest)
    print("%d-NN accuracy: %.3f" %(k, acc[i]))

plt.plot(k_list, acc)
plt.figure(1)


## Combination of the KNN
PRED = np.array([KNN[i].predict(xtest) for i in range(n_knn)])
predictions = np.array([get_max_prob(prob_dict(PRED[:,i])) for i in range(len(xtest))])
pred_proba = np.array([get_k_max_prob(prob_dict(PRED[:,i]), 2) for i in range(len(xtest))])
acc_test = np.sum(predictions==ytest)/len(ytest)
print("k-NN ensemble accuracy: %.3f" %acc_test)

#print(PRED[:,:10])
#print(predictions[:10] )

plt.figure(2)

#On redimensionne les données sous forme d'images
images = xtest.reshape((-1, 28, 28))
# On selectionne un echantillon d'images au hasard
select = np.random.randint(images.shape[0], size=45)

# On affiche les images avec la prédiction associée
for index, value in enumerate(select):
    plt.subplot(5,9,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('%s' % pred_proba[value])



plt.show()
