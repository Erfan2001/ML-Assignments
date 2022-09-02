import random 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 

np.random.seed(0)
centers=[[4,4], [-2, -1], [2, -3], [1, 1]]
X, Y = make_blobs(n_samples=5000, centers=centers, cluster_std=0.9)
# plt.scatter(X[:, 0], X[:, 1], marker='.')
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
print(k_means_cluster_centers,centers)
# for item in k_means_cluster_centers:
#     plt.scatter(item[0], item[1], marker='.',c="red")
# plt.legend(("point","centroid"))
plt.ylabel('Y')
plt.xlabel('X')

# *Initialize the plot with the specified dimensions*

colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
for k, col in zip(range(len(centers)), colors):
    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    # Plots the datapoints with color col.
    plt.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='black')
plt.show()
