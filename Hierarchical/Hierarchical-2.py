import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets ,preprocessing
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs
import scipy
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import euclidean_distances

def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )

pdf = pd.read_csv('Hierarchical/cars_clus.csv')[0:10]
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
pdf = pdf.replace("$null$",0)
featureSet = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
x = featureSet.values 
min_max_scaler = preprocessing.MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
leng = feature_mtx.shape[0]
#  *D & dist_matrix are the same (D => scipy | dist_matrix => scikit-learn)*
D = np.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
Z = hierarchy.linkage(D, 'complete')
# clusters = fcluster(Z, 5, criterion='maxclust')

fig, (ax1, ax2) = plt.subplots(2)
hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right',ax=ax1)
dist_matrix = euclidean_distances(feature_mtx,feature_mtx) 
Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')
hierarchy.dendrogram(Z_using_dist_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right',ax=ax2)
ax1.set_title("Scipy")
ax2.set_title("scikit-learn")
plt.show()
