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
from sklearn.linear_model import LogisticRegression
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


## Logistic Regression

if (SIMPLE_LOG_REG):
    print("\n===== SIMPLE LOGISTIC REGRESSION =====")

    lr = LogisticRegression(solver='liblinear')

    ## Training
    print("\n=== TRAINING ===")
    lr.fit(X_train,y_train)
    print("done")
    #print("Classes: ", lr.classes_)

    ## Test
    print("\n=== TEST ===")
    # Probability vector of being classed as 1 i.e. p
    y_prob = lr.predict_proba(X_test)[:,1] 
    # Make predictions
    y_pred = np.where(y_prob > 0.5, 1, 0) 

    # Look at some predictions
    y_test_sample=[int(i)-1 for i in np.linspace(1,len(y_prob),50)]
    print("Examine a sample of the test set")
    print("SAMPLE: \n", y_test_sample)
    print("\nPROBA of pred 1:\n", y_prob[y_test_sample])
    print("\nPREDICTIONS\n", y_pred[y_test_sample])
    y_test_array = np.array(y_test)
    print("\nY_TEST\n", y_test_array[y_test_sample])
    #print(y_pred[y_test_sample]+y_test_array[y_test_sample])
    

    ## ROC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("\nArea under ROC curve: {0:.3f}".format(roc_auc))

    # Plot ROC
    plt.figure()
    plt.title("Simple Logistic Regression\nReceiver Operating Characteristic")
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    #plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


if (CROSS_VAL_LOG_REG):
    print("\n===== CROSS VALIDATED LOGISTIC REGRESSION =====")
    
    # From the 2 below grid search, we find that l1 is the best penalty

    #lr = LogisticRegression(solver='liblinear',max_iter=1000)
    #params = { 'penalty':['l1','l2'] }
    # yields l1

    #lr = LogisticRegression(penalty='elasticnet',solver='saga',max_iter=5000)
    #params = { 'l1_ratio': np.linspace(0,1,5) }
    # yields l1_ratio = 1
    
    # Now find best penalty coefficient C

    #lr = LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000)
    #params = { 'C': np.logspace(-3, 6, 10) }
    # yields 100

    lr = LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000)
    params = { 'C': np.logspace(1, 2.8, 5) }
    # yields 223


    ## Training
    print("\n=== TRAINING ===")
    lr_gs = GridSearchCV(lr, params, cv=10)
    lr_gs.fit(X_train, y_train)
    #print("Classes: ", lr.classes_)
    print("Best parameters: ", lr_gs.best_params_)

    ## Test
    print("\n=== TEST ===")
    # Probability vector of being classed as 1 i.e. p
    y_prob = lr_gs.predict_proba(X_test)[:,1] 
    # Make predictions
    y_pred = np.where(y_prob > 0.5, 1, 0)

    ## ROC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("Area under ROC curve: {0:.3f}".format(roc_auc))

    # Plot ROC
    plt.figure()
    plt.title("Grid Search Cross Validation Logistic Regression\nReceiver Operating Characteristic")
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    #plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


plt.show()
