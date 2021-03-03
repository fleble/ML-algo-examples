#!/usr/bin/python3


########################## READ ME ##########################
#
# DESCRIPTION
# * Performance and predictions using a dummy classifier
# * Performance and predictions using a Linear SVC
# * Performance and predictions using a Non-Linear SVC
# * Comparison of different Non-Linear SVC: inspect the
#   Gram matrix to show feature of a good kernel
#
# RESULTS
# * On this dataset, the Linear SVC performs better than
#   a dummy classifier (better roc curve, higher roc_auc,
#   higher accuracy). The results are not so good though
#   and the Linear SVC does not converge during training,
#   indicating that the problem might not be linear.
# * On this dataset, the Non-Linear SVC performs better than
#   the Linear SVC: the corresponding problem seems rather
#   non-linear.
#
#############################################################


## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import model_selection, preprocessing, metrics
from sklearn import svm, dummy


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
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_class, train_size=0.8, random_state=RANDOM_STATE)
print("Train set size: {}".format(X_train.shape[0]))
print("Test set size : {}".format(X_test.shape[0]))


std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


## Training

score = 'roc_auc'

## DummyClassifier
print("\n===== Dummy Classifier (Baseline 1) =====")
dummy_clf = dummy.DummyClassifier(strategy='prior', random_state=RANDOM_STATE)
print("Training...")
dummy_clf.fit(X_train_std, y_train)

y_pred_proba_dummy = dummy_clf.predict_proba(X_test_std)[:, 1]
[fpr_dummy, tpr_dummy, thr_dummy] = metrics.roc_curve(y_test, y_pred_proba_dummy)
auc_dummy = metrics.auc(fpr_dummy,tpr_dummy)
acc_dummy = np.sum(dummy_clf.predict(X_test_std)==y_test)/len(y_test),
print("Prediction:")
print("\t%s = %0.3f" % (score, auc_dummy) )
print("\taccuracy = %0.3f" % acc_dummy)


print("\n===== Linear SVC (Baseline 2) =====")
# Create a linear SVC classifier with hyperparameter search using cross validation
param_grid = {'C': np.logspace(-2,2,5)}
clf = model_selection.GridSearchCV( svm.LinearSVC(max_iter=5000, random_state=RANDOM_STATE),
                                    param_grid,
                                    cv=5,
                                    scoring=score )

print("Training...")
clf.fit(X_train_std, y_train)

print("Best param: ", clf.best_params_)
print("Results of the cross validation:")
for mean, std, params in zip(
        clf.cv_results_['mean_test_score'], # mean score
        clf.cv_results_['std_test_score'],  # std score
        clf.cv_results_['params']           # hyperparameter value
        ):
    print("\t%s = %0.3f (+/-%0.3f) for %r" % (score, mean, 2*std, params))

y_test_pred = clf.decision_function(X_test_std)
[fpr, tpr, thr] = metrics.roc_curve(y_test, y_test_pred)
auc = metrics.auc(fpr,tpr)
acc = np.sum(clf.predict(X_test_std)==y_test)/len(y_test),
print("Prediction using best params:")
print("\t%s = %0.3f" %(score, auc))
print("\taccuracy = %0.3f" % acc)


print("\n===== Non-Linear SVC - RBF kernel =====")
# Create a non-linear SVC classifier with hyperparameter search using cross validation
#param_grid2 = { 'C': np.logspace(-2,2,5) ,'gamma': np.logspace(-2,2,5) }  # yields 1 1
param_grid2 = { 'gamma': np.logspace(-1,1,7) }
clf2 = model_selection.GridSearchCV( svm.SVC(kernel='rbf', random_state=RANDOM_STATE),
                                     param_grid2,
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

y_test_pred = clf2.decision_function(X_test_std)
[fpr2, tpr2, thr2] = metrics.roc_curve(y_test, y_test_pred)
auc2 = metrics.auc(fpr2,tpr2)
acc2 = np.sum(clf2.predict(X_test_std)==y_test)/len(y_test),
print("Prediction using best params:")
print("\t%s = %0.3f" %(score, auc2))
print("\taccuracy = %0.3f" % acc2)


# Create other non-linear SVC classifier with different gamma
gamma1=10**(-2)
print("\nCreate a non-linear SVC with gamma =",gamma1)
clf3 = svm.SVC(kernel='rbf', C=1, gamma=gamma1, random_state=RANDOM_STATE)

print("Training...")
clf3.fit(X_train_std, y_train)

y_test_pred = clf3.decision_function(X_test_std)
[fpr3, tpr3, thr3] = metrics.roc_curve(y_test, y_test_pred)
auc3 = metrics.auc(fpr3,tpr3)
acc3 = np.sum(clf3.predict(X_test_std)==y_test)/len(y_test),
print("Prediction:")
print("\t%s = %0.3f" %(score, auc3))
print("\taccuracy = %0.3f" % acc3)


gamma2=10**2
print("\nCreate a non-linear SVC with gamma =",gamma2)
clf4 = svm.SVC(kernel='rbf', C=1, gamma=gamma2, random_state=RANDOM_STATE)

print("Training...")
clf4.fit(X_train_std, y_train)

y_test_pred = clf4.decision_function(X_test_std)
[fpr4, tpr4, thr4] = metrics.roc_curve(y_test, y_test_pred)
auc4 = metrics.auc(fpr4,tpr4)
acc4 = np.sum(clf4.predict(X_test_std)==y_test)/len(y_test),
print("Prediction:")
print("\t%s = %0.3f" %(score, auc4))
print("\taccuracy = %0.3f" % acc4)



## Plot ROC
plt.figure()
plt.plot(fpr_dummy, tpr_dummy, color='red', label = 'Dummy clf ; AUC = %0.3f' % auc_dummy)
plt.plot(fpr, tpr, color='steelblue', label = 'Linear SVM ; AUC = %0.3f' % auc)
plt.plot(fpr2, tpr2, color='green', label = 'Non-Linear SVM gamma=%s ; AUC = %0.3f' % (clf2.best_params_['gamma'], auc2))
plt.plot(fpr3, tpr3, color='yellow', label = 'Non-Linear SVM gamma=%s ; AUC = %0.3f' % (gamma1, auc3))
plt.plot(fpr4, tpr4, color='orange', label = 'Non-Linear SVM gamma=%s ; AUC = %0.3f' % (gamma2, auc4))

plt.title("Linear vs Non-Linear SVM Classifier\nReceiver Operating Characteristic")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')


## Gram matrix:
kmatrix=[]
kmatrix.append( metrics.pairwise.rbf_kernel(X_train_std, gamma=clf2.best_params_['gamma']) )
kmatrix.append( metrics.pairwise.rbf_kernel(X_train_std, gamma=gamma1) )
kmatrix.append( metrics.pairwise.rbf_kernel(X_train_std, gamma=gamma2) )
GAM = [clf2.best_params_['gamma'], gamma1, gamma2]


fig, ax = plt.subplots(1,3)
fig.suptitle("Gram matrix\n")
size=100
for i in range(len(kmatrix)):
    axe=ax[i]
    axe.set_title("gamma="+str(GAM[i]))
    kmatrix100 = kmatrix[i][:size, :size]
    im=axe.pcolor(kmatrix100, cmap=cm.PuRd)  # draw matrix 
    
    axe.set_xlim([0, size])
    axe.set_ylim([0, size])
    axe.invert_yaxis()
    axe.xaxis.tick_top()

fig.subplots_adjust(top=0.8,right=0.8)
cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.65])
fig.colorbar(im, cax=cbar_ax)

plt.show()
