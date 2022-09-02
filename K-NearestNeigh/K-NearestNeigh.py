import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('K-NearestNeigh/teleCust1000t.csv')
# *Data Standardization gives the data zero mean and unit variance*
# Both X and X_same are the same
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
X_same=df.drop('custcat', axis=1).values
y = df['custcat'].values
NewX = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
neigh = KNeighborsClassifier(n_neighbors = 4).fit(X_train,y_train)
y_predicted = neigh.predict(X_test)
# Find Accuracy => print(metrics.accuracy_score(y_test, y_predicted))
# *Find best K*
Ks = 8
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    y_predicted=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_predicted)
    std_acc[n-1]=np.std(y_predicted==y_test)/np.sqrt(y_predicted.shape[0])

# Print best K => print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
#  *Visualization*
plt.plot(range(1,Ks),mean_acc,'g',color="blue")
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10,color="red")
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()