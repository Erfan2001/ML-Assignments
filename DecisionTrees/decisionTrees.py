import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

my_data = pd.read_csv("DecisionTrees/drug200.csv")
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
# Size => print(my_data.shape)
le_sex = preprocessing.LabelEncoder()
le_bp = preprocessing.LabelEncoder()
le_Cho = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
le_bp.fit(["LOW", "NORMAL", "HIGH"])
le_Cho.fit(['NORMAL', 'HIGH'])
# Convert to numerical for each feature
X[:, 1] = le_sex.transform(X[:, 1])
X[:, 2] = le_bp.transform(X[:, 2])
X[:, 3] = le_Cho.transform(X[:, 3])
Y = my_data["Drug"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=3)
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
drugTree.fit(X_train, Y_train)
predTree = drugTree.predict(X_test)
# Visualize Tree in Graph => tree.plot_tree(drugTree,filled=True)
# Visualize Tree in Text => print(tree.export_text(drugTree))
newDF = pd.DataFrame({'Actual': Y_test.values, 'Predicted': predTree})
accuracy = metrics.accuracy_score(Y_test, predTree)
# Accuracy => print(accuracy)
plt.show()
