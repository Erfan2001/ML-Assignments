import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

cust_df = pd.read_csv("K-Means/Cust_Segmentation.csv")
df = cust_df.drop('Address', axis=1)
X = df.values[:,1:]
X = np.nan_to_num(X)
standardize_dataset = StandardScaler().fit_transform(X)
# k_means_1 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
# k_means_1.fit(standardize_dataset)
# labels_1 = k_means_1.labels_
k_means_2 = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means_2.fit(X)
labels_2 = k_means_2.labels_
df["Clus_km"] = labels_2
divideByGroup=df.groupby('Clus_km').mean()

#  *Visualize 2D *
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], c=labels_2.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

#  *Visualize 3D *
# print(labels_2.astype(np.float))
# fig = plt.figure(1, figsize=(8, 6))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

# plt.cla()
# ax.set_xlabel('Education')
# ax.set_ylabel('Age')
# ax.set_zlabel('Income')

# ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels_2.astype(np.float))
plt.show()