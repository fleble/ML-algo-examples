#!/usr/bin/python3


######################
#   Load libraries
######################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

print("")


#######################
#   Define constants
#######################

RANDOM_SEED = rd.randint(0,10000)
RANDOM_STATE = RANDOM_SEED
print("RANDOM_STATE: ", RANDOM_STATE)

PLOT_NUMBER = 1


#################
#   Functions
#################

def plot_hyperplane(estimator, min_x, max_x, label, linestyle='k-'):
    # get the separating hyperplane
    # (eq) :  <w,x>+b=0  w=(x,y)
    #    <=>  w1*x + w2*y + b = 0
    #    <=>  y = ax + c  with a=-w1/w2 c=-b/w1
    w = estimator.coef_[0]
    a = -w[0] / w[1]
    x = np.linspace(min_x, max_x)
    y = a * x - (estimator.intercept_[0]) / w[1]
    plt.plot(x, y, linestyle, label=label)


###############
#    Main
###############

rd.seed(RANDOM_SEED)

n=[]
x1=[]
x2=[]

if (PLOT_NUMBER==1):
    N=4
    N_i = 500
    n.append( N_i )
    x1.append( [ x+0.5*rd.random() for x in np.linspace(2,8,n[0]) ] )
    x2.append( [ 3*x+rd.gauss(0,0.1*x*x) for x in x1[0] ] )

    n.append( N_i )
    x1.append( [ rd.gauss(3,1)  for i in range(n[1]) ] )
    x2.append( [ rd.gauss(22,3) for i in range(n[1]) ] )

    n.append( N_i )
    x1.append( [ x+0.5*rd.random() for x in np.linspace(-1,2,n[2]) ] )
    x2.append( [ -15*x+20+rd.gauss(0,1+1.5*x*x)+10*rd.random() for x in x1[2] ] )

    n.append( N_i )
    x1.append( [ 3+5*rd.random() for i in range(n[3]) ] )
    x2.append( [ -20+30*rd.random() for i in range(n[3]) ] )


if (PLOT_NUMBER==2):
    N=3
    N_i=500
    n.append( N_i )
    x1.append( [ x+0.5*rd.random() for x in np.linspace(2,8,n[0]) ] )
    x2.append( [ 3*x+rd.gauss(0,4) for x in x1[0] ] )

    n.append( N_i )
    x1.append( [ x+rd.gauss(0,1) for x in np.linspace(2,8,n[1]) ] )
    x2.append( [ 3*x+rd.gauss(0,4)+22 for x in x1[1] ] )

    n.append( N_i )
    x1.append( [ x+rd.random() for x in np.linspace(-1,1,n[2]) ] )
    x2.append( [ -20*x+rd.gauss(0,8)+22 for x in x1[2] ] )


if (PLOT_NUMBER==3):
    N=4
    N_i=800
    n.append( N_i )
    x1.append( [ x+0.5*rd.random() for x in np.linspace(2,8,n[0]) ] )
    x2.append( [ 3*x+rd.gauss(0,4) for x in x1[0] ] )

    n.append( N_i )
    x1.append( [ x+rd.gauss(0,1) for x in np.linspace(2,8,n[1]) ] )
    x2.append( [ 3*x+rd.gauss(0,4)+18 for x in x1[1] ] )

    n.append( N_i )
    x1.append( [ x+rd.gauss(0,1) for x in np.linspace(2,8,n[2]) ] )
    x2.append( [ 3*x+rd.gauss(0,4)+36 for x in x1[2] ] )
    
    n.append( N_i )
    x1.append( [ x+rd.random() for x in np.linspace(-2,1,n[3]) ] )
    x2.append( [ -20*x+rd.gauss(0,8)+22 for x in x1[3] ] )


x1_l=[]
x2_l=[]
for i in range(N): x1_l += x1[i]
for i in range(N): x2_l += x2[i]

color_l = []
for i in range(N): color_l += [i for j in range(n[i])]
color = ['b','r','g','y']

df = pd.DataFrame({'x1' : x1_l, 'x2' : x2_l, 'c' : color_l })
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

X = df.iloc[:,1:]
y = df.iloc[:,0]


## Check data
print("===== CHECK DATA =====")
print("\nRAW DATA")
print(df.head())
print("\nFEATURES")
print(X.head())
print("\nTARGET")
print(y.head())


## Split train and test
print("\n===== SPLIT TRAIN TEST =====")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=RANDOM_STATE)
print("Train set size: {}".format(X_train.shape[0]))
print("Test set size : {}".format(X_test.shape[0]))

x1_train_array = np.array(X_train.iloc[:,0])
x2_train_array = np.array(X_train.iloc[:,1])
y_train_array = np.array(y_train)
x1_test_array = np.array(X_test.iloc[:,0])
x2_test_array = np.array(X_test.iloc[:,1])
y_test_array = np.array(y_test)


## TRAINING
print("\n===== TRAINING =====")
SVM_ovr = OneVsRestClassifier(LinearSVC(max_iter=200000, random_state=RANDOM_STATE))
SVM_ovr.fit(X_train, y_train)

SVM_ovo = OneVsOneClassifier(LinearSVC(max_iter=200000, random_state=RANDOM_STATE))
SVM_ovo.fit(X_train, y_train)

y_pred_ovr = SVM_ovr.predict(X_test)
y_pred_ovo = SVM_ovo.predict(X_test)
#print("PREDICTIONS: \n",y_pred)
#print("\nY_TEST:\n", y_test_array)

print("\nOVR Accuracy: {0:.3f}".format( np.sum((y_pred_ovr==y_test_array))/y_test.shape[0] ))
print("OVO Accuracy: {0:.3f}".format( np.sum((y_pred_ovo==y_test_array))/y_test.shape[0] ))


## Plot distributions
            # 1      2      3
xmin=-3     # -2     -3     -3
xmax=10     # 10     10     10
ymin=-30    # -25    -30    -30
ymax=45     # 45     60     90

plt.figure()
for i in range(N): plt.plot(x1[i], x2[i], color=color[i], marker="o", linestyle='')
for i in range(N): plot_hyperplane(SVM_ovr.estimators_[i],xmin,xmax,"class"+str(i),color[i]+'-')

plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.title("One Versus Rest Classification")
plt.legend(loc='lower left')
plt.show()

