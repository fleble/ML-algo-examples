#!/usr/bin/python3

########################
##  Import libraries
########################
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics

########################
##  Define constants
########################
RANDOM_STATE = np.random.randint(10000)
print("RANDOM_STATE = %d" %RANDOM_STATE)

PLOT_NUMBER = 1



###################
##      MAIN
###################
np.random.seed(RANDOM_STATE)


##################################
##  Define a clustering problem
##################################
n=[]
x1=[]
x2=[]

if (PLOT_NUMBER==1):
    N=4
    N_i = 200
    n.append( N_i )
    x1.append( [ x+0.5*np.random.random() for x in np.linspace(2.5,8,n[0]) ] )
    x2.append( [ 3*x+np.random.normal(0,0.11*x*x) for x in x1[0] ] )

    n.append( N_i )
    x1.append( [ np.random.normal(2.9,1)  for i in range(n[1]) ] )
    x2.append( [ np.random.normal(35,3) for i in range(n[1]) ] )

    n.append( N_i )
    x1.append( [ x+0.5*np.random.random() for x in np.linspace(-1.2,1.8,n[2]) ] )
    x2.append( [ -15*x+20+np.random.normal(0,1+1.5*x*x)-15*np.random.random() for x in x1[2] ] )

    n.append( N_i )
    x1.append( [ 3+5*np.random.random() for i in range(n[3]) ] )
    x2.append( [ -30+30*np.random.random() for i in range(n[3]) ] )



# Put the data point to cluster in arrays
x1_l=[]
x2_l=[]
for i in range(N): x1_l += x1[i]
for i in range(N): x2_l += x2[i]
Npts = len(x1_l)

# Link them to a class (to evaluate the clustering)
color_l = []
for i in range(N): color_l += [i for j in range(n[i])]
color = ['b','r','g','y','orange','steelblue','black','coral']

# Shuffle the data
shuffler = [ i for i in range(Npts) ]
np.random.shuffle(shuffler)
for L in (x1_l, x2_l, color_l):
    L_tmp = L.copy()
    for i in range(Npts):
        L[i]=L_tmp[shuffler[i]]
    del L_tmp

# Define training points for clustering
X = np.array([x1_l,x2_l]).T
print("Data size: ",X.shape)

# Scale the data
std_scale = preprocessing.StandardScaler()
X_scaled = std_scale.fit_transform(X)


###################################
##  Train the cluster algorithms
###################################
n_clusters = 4

print("\n===== k-means =====")
kmeans1 = KMeans(n_clusters=n_clusters, n_init=10, max_iter=1000,
                 init='k-means++', random_state=RANDOM_STATE)

kmeans2 = KMeans(n_clusters=n_clusters, n_init=10, max_iter=1000,
                 init='k-means++', random_state=RANDOM_STATE)

print("* Unscaled data*")
print("Training...")
kmeans1.fit(X)
sil_score_unscaled = metrics.silhouette_score(X, kmeans1.labels_, metric='euclidean', random_state=RANDOM_STATE)
print("Silouhette score: %.3f" %sil_score_unscaled)

print("\n* Scaled data *")
print("Training...")
kmeans2.fit(X_scaled)
sil_score_scaled = metrics.silhouette_score(X_scaled, kmeans2.labels_, metric='euclidean', random_state=RANDOM_STATE)
print("Silouhette score: %.3f" %sil_score_scaled)


####################
##  Plot clusters
####################
#             1      2      3
xmin=-2     # -2     -3     -3
xmax=10     # 10     10     10
ymin=-40    # -25    -30    -30
ymax=55     # 45     60     90

xmin2=-2.5
xmax2=2.5
ymin2=-2.5
ymax2=2.5

'''
plt.figure()
for i in range(Npts): plt.plot(X[i,0], X[i,1], color=color[color_l[i]], marker="o", linestyle='')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.title("Originally designed classes - Unscaled data")
'''

plt.figure()
for i in range(Npts): plt.plot(X[i,0], X[i,1], color=color[kmeans1.labels_[i]], marker="o", linestyle='')
for i in range(n_clusters): plt.plot(kmeans1.cluster_centers_[i,0], kmeans1.cluster_centers_[i,1],
                            color='k', marker="s", linestyle='')
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.title("Classes found by k-means algorithm - Unscaled Data")

'''
plt.figure()
for i in range(Npts): plt.plot(X_scaled[i,0], X_scaled[i,1], color=color[color_l[i]], marker="o", linestyle='')
plt.xlim(xmin2,xmax2)
plt.ylim(ymin2,ymax2)
plt.title("Originally designed classes - Scaled data")
'''

plt.figure()
for i in range(Npts): plt.plot(X_scaled[i,0], X_scaled[i,1], color=color[kmeans2.labels_[i]], marker="o", linestyle='')
for i in range(n_clusters): plt.plot(kmeans2.cluster_centers_[i,0], kmeans2.cluster_centers_[i,1],
                            color='k', marker="s", linestyle='')
plt.xlim(xmin2,xmax2)
plt.ylim(ymin2,ymax2)
plt.title("Classes found by k-means algorithm - Scaled Data")


plt.show()

