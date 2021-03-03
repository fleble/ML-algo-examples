#!/usr/bin/python3


######################
##  LOAD LIBRAIRIES 
######################
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import utils.prod as upd

from sklearn.datasets import make_moons
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier 
from sklearn.linear_model import Perceptron
from sklearn.dummy import DummyClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



#################
##     MAIN 
#################

X, y = make_moons(n_samples=400, noise=0.25)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)


n_train=len(X_train)
X_train0=[]
X_train1=[]
for i in range(n_train):
    if (y_train[i]==0): X_train0.append(X_train[i])
    elif (y_train[i]==1): X_train1.append(X_train[i])
    else : print("error !")

X_train0=np.array(X_train0)
X_train1=np.array(X_train1)


## Bagging

print("===== Bagging =====")

# Perceptron grid
PERCEPTRON=[]
penalty_grid=['l1','l2']
alpha_grid=np.logspace(-2,2,5)
eta0_grid=np.logspace(-4,-1,4)

for p in upd.prod([['alpha',alpha_grid], ['eta0',eta0_grid]]):
    PERCEPTRON.append( Perceptron( penalty='l1',
                                   alpha=p['alpha'],
                                   eta0=p['eta0']  ) )

# K-nn grid
KNN = []
k_grid = [ i for i in range(3,8) ]
for p in upd.prod([['n_neighbors',k_grid]]):
    KNN.append( KNeighborsClassifier( n_neighbors=p['n_neighbors'] ) )

# Decision Tree
DT = []
criterion_grid = ['gini', 'entropy']
max_depth_grid = [None, 10, 20, 30, 50]
for p in upd.prod([ ['criterion', criterion_grid],
    ['max_depth', max_depth_grid] ]):
    DT.append( DecisionTreeClassifier( criterion=p['criterion'],
                                       max_depth=p['max_depth'] ) )


# SVC
SVMC = []
C_grid = np.logspace(-1,2,4)
gamma_grid = np.logspace(-1,2,4)
for p in upd.prod([['C',C_grid], ['gamma',gamma_grid]]):
    SVMC.append( SVC( C=p['C'],
                 gamma=p['gamma'] ) )


## Training
score='accuracy'
param_grid={ 'base_estimator': [ DummyClassifier()] +
                               PERCEPTRON +
                               DT +
                               KNN +
                               SVMC }

bagging = model_selection.GridSearchCV( BaggingClassifier(n_estimators=100),
                                        param_grid = param_grid,
                                        scoring = score,
                                        cv = 4 )

print("Training...")
bagging.fit(X_train, y_train)


print("Best param: ", bagging.best_params_)
print("Results of the cross validation:")
for mean, std, params in zip(
        bagging.cv_results_['mean_test_score'],
        bagging.cv_results_['std_test_score'],
        bagging.cv_results_['params']
        ):
    print("\n\n\t%s = %0.3f (+/-%0.3f) for \n\t%r" % (score, mean, 2*std, params))


print("\n\nPrediction using best params:")
print("%s = %0.3f for \n%r" % (score, np.sum(bagging.predict(X_test)==y_test)/len(y_test),bagging.best_params_))


print("\n\n===== Use the best method on its own =====")

best_clf = bagging.best_params_['base_estimator']
print("Training...")
best_clf.fit(X_train,y_train)

print("Prediction:")
print("%s = %.3f for" %(score, np.sum(best_clf.predict(X_test)==y_test)/len(y_test) ) )
print(best_clf)

plt.figure()
plt.plot(X_train0[:, 0], X_train0[:, 1], 'ro')
plt.plot(X_train1[:, 0], X_train1[:, 1], 'bo')
plt.show()
