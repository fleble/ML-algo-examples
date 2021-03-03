#!/usr/bin/python3


########################## READ ME ##########################
#
# DESCRIPTION
# * Algorithm for a kNN classifier with grid search using
#   cross validation
# * Results of the cross validation are printed out (mean
#   accuracy, standard deviation) for the different
#   hyperparameter values.
# * Prediction using the test set and accuracy
# * ROC curve and area under it (auc)
# * Comparison with dummy classifier
#
# RESULTS
# * Features are on very different scales. To scale or not
#   to scale the features data is investigated.
#   Scaling yields better results.
# * When enabling grid search with k=1, k=1 is always chosen.
#   This yields better accuracy, but smaller auc. Choosing
#   k=1,2 should never made be possible.
#
#############################################################


## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn import neighbors, metrics
from sklearn.dummy import DummyClassifier

print("")


## Define constants
RANDOM_STATE = np.random.randint(10000)
print("random_state = ",RANDOM_STATE)


## Read data
data = pd.read_csv('./input/winequality-white.csv', sep=";")
# Shuffle
data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
# Target
y_class = np.where(y<6, 0, 1)


## Check data
print("\n===== CHECK DATA =====")
print("\nRAW DATA")
print(data.head())
print("\nFEATURES")
print(X.head())
print("\nTARGET")
print(y_class[:5])


## Split train and test
print("\n===== SPLIT TRAIN TEST =====")
X_train, X_test, y_train, y_test = train_test_split(X, y_class, train_size=0.8, random_state=RANDOM_STATE)
print("Train set size: {}".format(X_train.shape[0]))
print("Test set size : {}".format(X_test.shape[0]))


std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


## Training
# Fix hyperparameters values to be tested
param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15, 20, 25]}
score = 'accuracy'

# For non scaled data:
print("\n===== Unscaled data =====")

# Create a kNN classifier with hyperparameter search using cross validation
clf = GridSearchCV(neighbors.KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring=score
        )

print("Training...")
clf.fit(X_train, y_train)

print("Best param: ", clf.best_params_)
print("Results of the cross validation:")
for mean, std, params in zip(clf.cv_results_['mean_test_score'], # score moyen
        clf.cv_results_['std_test_score'], # écart-type du score
        clf.cv_results_['params'] # valeur de l'hyperparamètre
        ):
    print("\t%s = %0.3f (+/-%0.3f) for %r" % (score, # critère utilisé
            mean, # score moyen
            std * 2, # barre d'erreur
            params # hyperparamètre
            ))

print("Prediction using best params:")
print("%s = %0.3f for %r" % (score,
                             np.sum(clf.predict(X_test)==y_test)/len(y_test),
                             clf.best_params_
                             ))

y_pred_proba = clf.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.auc(fpr,tpr)
print("Area under ROC curve: %0.3f"%auc)


# For scaled data:
print("\n===== Scaled data =====")

clf2 = GridSearchCV( neighbors.KNeighborsClassifier(),
                     param_grid,
                     cv=5,
                     scoring=score )

print("Training...")
clf2.fit(X_train_std, y_train)

print("Best param: ", clf2.best_params_)
print("Results of the cross validation:")
for mean, std, params in zip(
        clf2.cv_results_['mean_test_score'], # mean score
        clf2.cv_results_['std_test_score'],  # std score
        clf2.cv_results_['params']           # hyperparameter value
        ):
    print("\t%s = %0.3f (+/-%0.3f) for %r" % (score, mean, 2*std, params))

print("Prediction using best params:")
print("\t%s = %0.3f for %r" % (score,
                               np.sum(clf2.predict(X_test_std)==y_test)/len(y_test),
                               clf2.best_params_ ))

y_pred_proba2 = clf2.predict_proba(X_test_std)[:, 1]
[fpr2, tpr2, thr2] = metrics.roc_curve(y_test, y_pred_proba2)
auc2 = metrics.auc(fpr2,tpr2)
print("Area under ROC curve: %0.3f"%auc2)


## DummyClassifier
print("\n===== Dummy Classifier =====")
dummy_clf = DummyClassifier(strategy='prior', random_state=RANDOM_STATE)
print("Training...")
dummy_clf.fit(X_train, y_train)

print("Prediction:")
print("%s = %0.3f" % (score,
                      np.sum(dummy_clf.predict(X_test)==y_test)/len(y_test),
                     ))


y_pred_proba_dummy = dummy_clf.predict_proba(X_test)[:, 1]
[fpr_dummy, tpr_dummy, thr_dummy] = metrics.roc_curve(y_test, y_pred_proba_dummy)
auc_dummy = metrics.auc(fpr_dummy,tpr_dummy)
print("Area under ROC curve: %0.3f"%auc_dummy)


## Plot ROC curve
plt.plot(fpr, tpr, color='steelblue', lw=2, label="unscaled data; auc=%0.3f"%auc)
plt.plot(fpr2, tpr2, color='red', lw=2,     label="scaled data    ; auc=%0.3f"%auc2)
plt.plot(fpr_dummy, tpr_dummy, color='green', lw=2, label="dummy clf      ; auc=%0.3f"%auc_dummy)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - specificity', fontsize=10)
plt.ylabel('Sensitivity', fontsize=10)
plt.legend(loc='lower right')
plt.show()
