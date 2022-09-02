import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, f1_score,log_loss


cell_df = pd.read_csv("SVM/cell_samples.csv")
ax = cell_df[cell_df['Class'] == 4][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant', s=50)
cell_df[cell_df['Class'] == 2][0:50].plot(
    kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax)
# plt.show()
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
feature_df = cell_df.drop('Class', axis=1).values
X = np.asarray(feature_df)
cell_df['Class'] = cell_df['Class'].astype('int')
Y = np.asarray(cell_df['Class'])
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=4)

kernels = ['rbf', 'poly', 'sigmoid']
for item in kernels:
    clf = svm.SVC(kernel=item)
    clf.fit(X_train, Y_train)
    Y_predicted = clf.predict(X_test)
    cnf_matrix = confusion_matrix(Y_test, Y_predicted, labels=[2, 4])
    # np.set_printoptions(precision=2)
    # report=classification_report(Y_test, Y_predicted)
    f1Score = f1_score(Y_test, Y_predicted, average='weighted')
    jaccard = jaccard_score(Y_test, Y_predicted, average='weighted')
    logLoss=log_loss(Y_test, Y_predicted)
    print(cnf_matrix)
    print(f1Score)
    print(jaccard)
    print(logLoss)
    print("------------")
