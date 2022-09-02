import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets ,preprocessing
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs

np.random.seed(0)
centers=[[4,4], [-2, -1], [1, 1], [10,4]]
X, Y = make_blobs(n_samples=500, centers=centers, cluster_std=0.9)
# plt.scatter(X[:, 0], X[:, 1], marker='o') 
agglomerative = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
agglomerative.fit(X,Y)
Z = preprocessing.StandardScaler().fit(X).transform(X)
# Create a minimum and maximum range of X.
x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
labels=agglomerative.labels_
# Get the average distance for X.
X = (X - x_min) / (x_max - x_min)
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
fig, (ax1, ax2) = plt.subplots(2)
# This loop displays all of the dataPoints:
for k , col in zip(range(len(centers)), colors):
    my_members = (labels == k)
    ax1.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax2.plot(Z[my_members, 0], Z[my_members, 1], 'w', markerfacecolor=col, marker='.')
ax1.set_title("Average Distance for X")
ax2.set_title("Using StandardScaler")
# plt.show()

dist_matrix = distance_matrix(X,X) 
completeLinkage = hierarchy.linkage(dist_matrix, 'complete')
dendrogram = hierarchy.dendrogram(completeLinkage)
print(dendrogram)