#!/usr/bin/python3


########################
### Import libraries ###
########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


###############################
### Define useful constants ###
###############################

# tts 0.5: 4373 5340
# tts 0.8: 7589
RANDOM_STATE = 5340#rd.randint(0,10000)
print("RANDOM_STATE = ",RANDOM_STATE)
STANDARD_SCALING = 1


############################
########### MAIN ###########
############################

## Read data
raw_data = pd.read_csv('./input/prostate_dataset.txt', delimiter='\t')
X = raw_data.iloc[:,1:-2]
y = raw_data.iloc[:,-2]
X_cols = X.columns


## Data size
print("\n===== DATA SIZE =====")
print("Raw data: ",raw_data.shape)

## Check data
print("\n===== CHECK DATA =====")
print("RAW DATA")
print(raw_data.head())
print("\n")
print("FEATURES")
print(X.head())
print("\n")
print("TARGET")
print(y.head())

## Standard scaling for X
if (STANDARD_SCALING):
    print(" \n===== Standard scaling for X =====")
    std_scaling = preprocessing.StandardScaler().fit(X)
    X = pd.DataFrame(std_scaling.transform(X))
    # Columns names lost during scaling, redefine them
    X.columns=X_cols
    print(X.head())


## Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=RANDOM_STATE)
'''
X_train = raw_data.iloc[31:-1,1:-3]
y_train = raw_data.iloc[31:-1,-2]
X_test = raw_data.iloc[:30,1:-3]
y_test = raw_data.iloc[:30,-2]
'''

## Linear Regression - Baseline

print("\n===== Linear Regression =====")
lr = LinearRegression()
lr.fit(X_train,y_train)
baseline_error = np.mean((lr.predict(X_test) - y_test) ** 2)
print("Linear Regression baseline error: {0:.2f}".format(baseline_error))


## Ridge Regression

print("\n===== Ridge Regression =====")
n_alphas = 200
alphas = np.logspace(-5, 5, n_alphas)
ridge = Ridge()

coefs = []
errors = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)
    errors.append(np.mean((ridge.predict(X_test) - y_test) ** 2))


fig,ax=plt.subplots(2,1,sharex=True)
fig.subplots_adjust(hspace=0)

coefs=np.array(coefs) # convert list into np.array
for i in range(len(coefs[0])):
    ax[0].plot(alphas, coefs[:,i], label="c"+str(i))
ax[1].plot(alphas, errors, label="Ridge Regression")
ax[1].plot([alphas[0],alphas[-1]], [baseline_error, baseline_error], label="Standard Linear Regression")
ax[0].set_title("Ridge Regression:\nCoefficients - MSE")
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[1].set_xlabel('alpha')
ax[0].set_ylabel('Weights')
ax[1].set_ylabel("Mean Square Error")
ax[0].legend()
ax[1].legend()


argmin = np.argmin(errors)
alpha_best = alphas[argmin]
error_min = errors[argmin]
print("Best hyperparameter alpha: {0:.2f}".format(alpha_best))
print("Minimal error: {0:.2f}".format(error_min))


plt.show()
