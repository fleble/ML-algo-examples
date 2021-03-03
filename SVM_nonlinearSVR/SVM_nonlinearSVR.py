#!/usr/bin/python3


########################## READ ME ##########################
#
# DESCRIPTION
# *
#
# RESULTS
# * 
#
#############################################################


## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import model_selection, preprocessing, metrics
from sklearn import dummy, linear_model, svm, kernel_ridge


## Define constants
RANDOM_STATE = np.random.randint(10000)
print("random_state = ",RANDOM_STATE)


dum  = 1
lRR  = 1
kRR  = 1
lSVR = 1
SVR  = 1

## Read data
data = pd.read_csv('./input/winequality-white.csv', sep=";")
# Shuffle
data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
X = data.iloc[:,:-1]
y = data.iloc[:,-1]


## Check data
print("\n===== CHECK DATA =====")
print("\nRAW DATA")
print(data.head())
print("\nFEATURES")
print(X.head())
print("\nTARGET")
print(y.head())


## Split train and test
print("\n===== SPLIT TRAIN TEST =====")
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, random_state=RANDOM_STATE)
print("Train set size: {}".format(X_train.shape[0]))
print("Test set size : {}".format(X_test.shape[0]))


std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


## Gram matrix:
kmatrix=[]
subtitles=[]


## Training
score = 'neg_mean_squared_error'


## DummyClassifier
if (dum):
    print("\n===== Dummy Classifier (Baseline 1) =====")
    rgs_dum = dummy.DummyRegressor(strategy='mean')

    print("Training...")
    rgs_dum.fit(X_train_std, y_train)

    print("Prediction:")
    y_test_pred = rgs_dum.predict(X_test_std)
    rmse_dum = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    print("\tRMSE = %0.3f" % rmse_dum)


## Linear Ridge Regressor
if (lRR):
    print("\n===== Linear Ridge Regressor =====")
    param_grid = { 'alpha': np.logspace(-3, 3,7) }
    rgs_lrr = model_selection.GridSearchCV( linear_model.Ridge(),
                                            param_grid=param_grid,
                                            scoring=score,
                                            cv=4  )

    print("Training...")
    rgs_lrr.fit(X_train_std, y_train)

    print("Best param: ", rgs_lrr.best_params_) 
    print("Results of the cross validation:")
    for mean, std, params in zip(
                rgs_lrr.cv_results_['mean_test_score'], # mean score
                rgs_lrr.cv_results_['std_test_score'],  # std score
                rgs_lrr.cv_results_['params']           # hyperparameter value
                ):
        print("\t%s = %0.3f (+/-%0.3f) for %r" % (score, mean, 2*std, params))

    print("Prediction using best params:")
    y_test_pred = rgs_lrr.predict(X_test_std)
    rmse_lrr = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    print("\tMSE : %.3f" % rmse_lrr**2)
    print("\tRMSE: %.3f" % rmse_lrr)


## kernel Ridge Regressor
if (kRR):
    print("\n===== kRR (kernel Ridge Regressor) - RBF kernel =====")
    param_grid = { 'alpha': np.logspace(-3,-1,3), 'gamma': np.logspace(-3,-1,3) }
    rgs_krr = model_selection.GridSearchCV( kernel_ridge.KernelRidge(kernel='rbf'),
                                    param_grid=param_grid,
                                    scoring=score,
                                    cv=4  )

    print("Training...")
    rgs_krr.fit(X_train_std, y_train)

    print("Best param: ", rgs_krr.best_params_) 
    print("Results of the cross validation:")
    for mean, std, params in zip(
                rgs_krr.cv_results_['mean_test_score'], # mean score
                rgs_krr.cv_results_['std_test_score'],  # std score
                rgs_krr.cv_results_['params']           # hyperparameter value
                ):
        print("\t%s = %0.3f (+/-%0.3f) for %r" % (score, mean, 2*std, params))

    print("Prediction using best params:")
    y_test_pred = rgs_krr.predict(X_test_std)
    rmse_krr = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    print("\tMSE : %.3f" % rmse_krr**2)
    print("\tRMSE: %.3f" % rmse_krr)

    kmatrix.append( metrics.pairwise.rbf_kernel(X_train_std, gamma=rgs_krr.best_params_['gamma']) )
    subtitles.append( 'kRR - gamma ='+str(rgs_krr.best_params_['gamma']) )



## Linear SVR
if (lSVR):
    print("\n===== Linear SVR =====")
    # Create a linear SVM Regressor with hyperparameter search using cross validation
    param_grid = { 'C': np.logspace(-2,2,5) }
    rgs_lsvr = model_selection.GridSearchCV( svm.LinearSVR(max_iter=10000),
                                             param_grid = param_grid,
                                             scoring=score,
                                             cv=4 )

    print("Training...")
    rgs_lsvr.fit(X_train_std, y_train)

    print("Best param: ", rgs_lsvr.best_params_) 
    print("Results of the cross validation:")
    for mean, std, params in zip(
            rgs_lsvr.cv_results_['mean_test_score'], # mean score
            rgs_lsvr.cv_results_['std_test_score'],  # std score
            rgs_lsvr.cv_results_['params']           # hyperparameter value
            ):
        print("\t%s = %0.3f (+/-%0.3f) for %r" % (score, mean, 2*std, params))

    print("Prediction using best params:")
    y_test_pred = rgs_lsvr.predict(X_test_std)
    rmse_lsvr = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    print("\tMSE : %.3f" % rmse_lsvr**2)
    print("\tRMSE: %.3f" % rmse_lsvr)


## SVR
if (SVR):
    print("\n===== Non-Linear SVR - RBF kernel =====")
    # Create a non-linear SVM Regressor with hyperparameter search using cross validation
    param_grid2 = { 'C': np.logspace(-1,1,3) ,'gamma': np.logspace(0,2,3) }
    rgs_svr = model_selection.GridSearchCV( svm.SVR(kernel='rbf'),
                                            param_grid = param_grid2,
                                            scoring=score,
                                            cv=4 )

    print("Training...")
    rgs_svr.fit(X_train_std, y_train)

    print("Best param: ", rgs_svr.best_params_) 
    print("Results of the cross validation:")
    for mean, std, params in zip(
            rgs_svr.cv_results_['mean_test_score'], # mean score
            rgs_svr.cv_results_['std_test_score'],  # std score
            rgs_svr.cv_results_['params']           # hyperparameter value
            ):
        print("\t%s = %0.3f (+/-%0.3f) for %r" % (score, mean, 2*std, params))

    print("Prediction using best params:")
    y_test_pred = rgs_svr.predict(X_test_std)
    rmse_svr = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    print("\tMSE : %.3f" % rmse_svr**2)
    print("\tRMSE: %.3f" % rmse_svr)

    kmatrix.append( metrics.pairwise.rbf_kernel(X_train_std, gamma=rgs_svr.best_params_['gamma']) )
    subtitles.append( 'SVR - gamma ='+str(rgs_svr.best_params_['gamma']) )


## Gram matrix:
n=len(kmatrix)

if (n):
    fig, ax = plt.subplots(1,n)
    fig.suptitle("Gram matrix\n")
    size=100
    im=None
    for i in range(n):
        axe=ax[i]
        axe.set_title(subtitles[i])
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

