import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
import warnings
from sklearn.cluster import KMeans 

np.random.seed(150)

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    # Create random data and store in feature matrix X and response vector y.
    X, Y = make_blobs(n_samples=numSamples, centers=centroidLocation, 
                                cluster_std=clusterDeviation)
    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)
    return X, Y
X, Y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)
epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)

fig, (ax1, ax2) = plt.subplots(2)
# *DBSCAN*
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    if k == -1:
        # Black used for noise.
        col = 'k'
    # Plot the dataPoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    ax1.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker='.', alpha=0.8)
    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    ax1.scatter(xy[:, 0], xy[:, 1],s=50, c=[col], marker='.', alpha=0.8)
# *K-Means*
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means.fit(X)
k_means_labels = k_means.labels_
for k, col in zip(k_means_labels, colors):
    my_members = (k_means_labels == k)
    ax2.scatter(X[my_members, 0], X[my_members, 1], c=col,marker='.', alpha=0.8)
ax1.set_title("DBSCAN")
ax2.set_title("K-Means")
plt.show()