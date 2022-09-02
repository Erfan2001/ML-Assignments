import numpy as np
import pandas as pd
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score,confusion_matrix,classification_report,log_loss,precision_score,recall_score
import matplotlib.pyplot as plt

churn_df = pd.read_csv('LogisticReg/ChurnData.csv')
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
Y = np.asarray(churn_df['churn'])
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)
Y_predicted = LR.predict(X_test)
# *predict proba*
Y_prob = LR.predict_proba(X_test)
# *jaccard index*
ja=jaccard_score(Y_test, Y_predicted,pos_label=0)
acc=metrics.accuracy_score(Y_test, Y_predicted)
conf=confusion_matrix(Y_test, Y_predicted, labels=[1,0])
report=classification_report(Y_test, Y_predicted)
logLass=log_loss(Y_test, Y_prob)
precision=precision_score(Y_test, Y_predicted)
recall=recall_score(Y_test, Y_predicted)
print (report)
print(logLass)
print(precision)
print(recall)
print(conf)