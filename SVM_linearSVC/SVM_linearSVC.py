#!/usr/bin/python3

########################
### Import libraries ###
########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc


print("") # the first printout never works
          # so print dummy string here

###############################
### Define useful constants ###
###############################

RANDOM_STATE = rd.randint(0,10000)
print("RANDOM_STATE = {}".format(RANDOM_STATE))

SIMPLE_LOG_REG    = 1
CROSS_VAL_LOG_REG = 1


############################
########### MAIN ###########
############################

## Read data
raw_data = pd.read_csv('./input/mushrooms_dataset.csv')

print("===== RAW DATA BEFORE LABEL ENCODING =====")
print(raw_data.head())


## Label encoding
labelencoder=LabelEncoder()
for col in raw_data.columns:
    raw_data[col] = labelencoder.fit_transform(raw_data[col])


X = raw_data.iloc[:,1:23]
y = raw_data.iloc[:,0]

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


## Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


## SVM Linear Classifier

# Training
print("\n===== TRAINING =====")
svm = LinearSVC(penalty="l2", dual=False, max_iter=10000)
params = { 'C': np.logspace(-3, 4, 16) }

gs_svm = GridSearchCV(svm, params, cv=10)
gs_svm.fit(X_train, y_train)

print("Best params: ", gs_svm.best_params_)


# Prediction
y_prob = gs_svm.predict(X_test)
#print(y_prob)


# ROC
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("\nArea under ROC curve: {0:.3f}".format(roc_auc))

# Plot ROC
plt.figure()
plt.title("Grid Search Cross Validation SVM Classifier\nReceiver Operating Characteristic")
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
#plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()
