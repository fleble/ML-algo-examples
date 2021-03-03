#!/usr/bin/python3


############################## READ ME ##############################
#
# DESCRIPTION
# * DBSCAN to do clustering on different type of data: same density
#   or different density data (PLOT_NUMBER)
# * OPTICS to do clustering on the same data as previous point
# * OPTICS best hyperparameters can be found using a grid search
#   or by hand (GRID_SEARCH)
#
# RESULTS
# * As expected, DBSCAN is very efficient when clusters have the
#   same density
# * As expected too, DBSCAN fails when clusters densities are
#   different: will either merge high density clusters or treat
#   low density clusters as noise
# * OPTICS can help find clusters having different densities.
#   However, finfing good hyperparameters turns out to be a hard
#   task. Grid search and maximisation of the silouhette score
#   usually yields accurate results :)
# * On this set of examples, it seems that the best score to
#   evaluate the model is:
#      n_clusters**A * silhouette_score / log(n_noise)
#   where n_clusters      is the number of clusters
#         A                  a constant beetween 1.2 and 1.5
#         silouette_score    the well-known silhouette score
#         n_noise            the number of noise points
#   When n_noise = 0 or 1, we use log(2) instead of log(n_noise)
#   This maximises the number of clusters (n_clusters**A),
#   minimises huge numbers of noise points (log(n_noise) so that
#   clusters are not forgotten), and provides reasonable results
#   thanks to silhouette_score.
#
#####################################################################


## Import libraries

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import utils.prod as upd


RANDOM_STATE = np.random.randint(10000)
print("RANDOM_STATE = %d" %RANDOM_STATE)

PLOT_NUMBER = 3  # 1  2  3
PLOT_TEXT = ''   # 'Same density case\n' 'Different densities case\n'  ''
GRID_SEARCH_DBSCAN = 1
GRID_SEARCH_OPTICS = 1

np.random.seed(RANDOM_STATE)


## #############################################################################
## Generate sample data
# Same density
if (PLOT_NUMBER==1):
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                    random_state=0)

# Different densities
if (PLOT_NUMBER==2):
    centers_l = [ [-1, -1], [-1, 1] ]
    centers_r = [ [1, -1], [1, 1] ]
    X_l, labels_true_l = make_blobs(n_samples=500, centers=centers_l, cluster_std=0.4,
                                    random_state=RANDOM_STATE)
    X_r, labels_true_r = make_blobs(n_samples=100, centers=centers_r, cluster_std=0.4,
                                    random_state=RANDOM_STATE)
    X=np.concatenate((X_l,X_r))
    labels_true=np.concatenate((labels_true_l, labels_true_r))

# Other kind
if (PLOT_NUMBER==3):
    n=[]
    x1=[]
    x2=[]
    N=4
    N_i = 200
    n.append( 400 )
    x1.append( [ x+0.5*np.random.random() for x in np.linspace(2.5,8,n[0]) ] )
    x2.append( [ 3*x+np.random.normal(0,0.11*x*x) for x in x1[0] ] )

    n.append( 300 )
    x1.append( [ np.random.normal(2.9,1)  for i in range(n[1]) ] )
    x2.append( [ np.random.normal(35,3) for i in range(n[1]) ] )

    n.append( 200 )
    x1.append( [ x+0.5*np.random.random() for x in np.linspace(-1.2,1.8,n[2]) ] )
    x2.append( [ -15*x+20+np.random.normal(0,1+1.5*x*x)-15*np.random.random() for x in x1[2] ] )

    n.append( 100 )
    x1.append( [ 3+5*np.random.random() for i in range(n[3]) ] )
    x2.append( [ -30+30*np.random.random() for i in range(n[3]) ] )

    # Put the data point in array
    x1_l=[]
    x2_l=[]
    for i in range(N): x1_l += x1[i]
    for i in range(N): x2_l += x2[i]
    labels_true = []
    for i in range(N): labels_true += [i for j in range(n[i])]

    # Shuffle the data
    # This is necessary for OPTICS to provide good results ! (don't know why though)
    shuffler = [ i for i in range(sum(n)) ]
    np.random.shuffle(shuffler)
    for L in (x1_l, x2_l, labels_true):
        L_tmp = L.copy()
        for i in range(sum(n)): L[i]=L_tmp[shuffler[i]]
        del L_tmp
    
    X=np.array([x1_l, x2_l]).T


X = StandardScaler().fit_transform(X)

## #############################################################################
## Compute DBSCAN
print("===== DBSCAN =====")

# Find best hyperparameters using a grid search
if (GRID_SEARCH_DBSCAN == 1):
    print("Find best eps and min_samples by grid search and maximisation of silhouette_score**ALPHA*n_clusters/log(n_noise)")
    print("The trick to divide by log(n_noise) aims to reject clustering having a good silhouette_score but missing an entire cluster !")
    print("The trick to multiply by n_cluster**ALPHA (ALPHA~1..1.5) is to favour high number of cluster rather than merging small clusters together. Too high number of clusters is deeply unfavoured due a bad silhouette score.")
    models = []
    models_score = []
    min_samples_grid = [5,10,20,30,40,45,50]
    eps_grid = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
    param_grid = upd.prod([['min_samples',min_samples_grid], ['eps',eps_grid]])
    for p in param_grid:
        models.append( DBSCAN( min_samples=p['min_samples'], eps=p['eps'],
                               metric='minkowski', p=2 ) )

    print("Training...")
    ALPHA=0.5
    print("   Params \t\t\t\tScore \t\tn_clusters")
    for i,m in enumerate(models):
        m.fit(X)
        try:
            labels = m.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            if (n_noise>=2): log_n_noise=np.log(n_noise)
            else: log_n_noise=np.log(2)
            models_score.append(n_clusters**ALPHA*metrics.silhouette_score(X, labels)/log_n_noise)
            print("   %s:\t%.3f \t\t%d" %(param_grid[i], models_score[i], n_clusters))
        except:
            print("   %s:\tNA" %(param_grid[i]))
            models_score.append(-2)
 
    best_model_idx = np.argmax(models_score)
    best_params = param_grid[best_model_idx]
    print("\nBest param: %s" %best_params)
    db = models[best_model_idx]
    

# Try a set of hyperparameters
if (GRID_SEARCH_DBSCAN != 1):
    print("Training...")
    db = DBSCAN(eps=0.5, min_samples=40, metric='minkowski', p=2).fit(X)  # 2 : 0.5, 40


core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


## #############################################################################
## Plot result
plt.figure(1)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('%sDBSCAN - Estimated number of clusters: %d' % (PLOT_TEXT, n_clusters_))



## #############################################################################
## Compute OPTICS
print("\n===== OPTICS =====")

# Find best hyperparameters using a grid search
if (GRID_SEARCH_OPTICS == 1):
    print("Find best xi and min_samples by grid search and maximisation of silhouette_score**ALPHA*n_clusters/log(n_noise)")
    print("The trick to divide by log(n_noise) aims to reject clustering having a good silhouette_score but missing an entire cluster ! This does happend for maximisation of silhouette_score in case 1, yielding min_samples=0.03 xi=0.05 but missing the whole top right cluster")
    print("The trick to multiply by n_cluster**ALPHA (ALPHA~1..1.5) is to favour high number of cluster rather than merging small clusters together. Too high number of clusters is deeply unfavoured due a bad silhouette score.")
    models = []
    models_score = []
    min_samples_grid = [10,20,30,40,45]
    xi_grid = [0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.065, 0.08, 0.1]
    param_grid = upd.prod([['min_samples',min_samples_grid], ['xi',xi_grid]])
    for p in param_grid:
        models.append( OPTICS( min_samples=p['min_samples'], xi=p['xi'], cluster_method='xi',
                               min_cluster_size=0.05, max_eps=np.inf, metric='minkowski', p=2 ) )

    print("Training...")
    ALPHA=0.5
    print("   Params \t\t\t\tScore \t\tn_clusters")
    for i,m in enumerate(models):
        m.fit(X)
        try:
            labels = m.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            if (n_noise>=2): log_n_noise=np.log(n_noise)
            else: log_n_noise=np.log(2)
            models_score.append(n_clusters**ALPHA*metrics.silhouette_score(X, labels)/log_n_noise)
            print("   %s:\t%.3f \t\t%d" %(param_grid[i], models_score[i], n_clusters))
        except:
            print("   %s:\tNA" %(param_grid[i]))
            models_score.append(-2)
        
    best_model_idx = np.argmax(models_score)
    best_params = param_grid[best_model_idx]
    print("\nBest param: %s" %best_params)
    optics = models[best_model_idx]

# Try a set of hyperparameters
if (GRID_SEARCH_OPTICS != 1):
    optics = OPTICS( min_samples=40, cluster_method='xi', xi=0.03,        # 2: 30, 0.05
                     min_cluster_size=.05, max_eps=np.inf, metric='minkowski', p=2 )

    print("Training...")
    optics.fit(X)


# Compute model quality quantities and model features
# Number of clusters in labels, ignoring noise if present.
labels = optics.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
#print("\nREACH\n",optics.reachability_)
#print("\nORDER\n",optics.ordering_)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
         % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))


## #############################################################################
## Plot result
plt.figure(2)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    
    #xy = X[class_member_mask & core_samples_mask]
    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #         markeredgecolor='k', markersize=10)

    #xy = X[class_member_mask & ~core_samples_mask]
    #plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #         markeredgecolor='k', markersize=6)
    
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('%sOPTICS - Estimated number of clusters: %d' % (PLOT_TEXT, n_clusters_))


plt.show()

