import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# *Show bar chart*
df = pd.read_csv('Exam/loan_train.csv')
df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
row="age"
bins = np.linspace(df[row].min(), df[row].max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, row, bins=bins, ec="k")
g.axes[-1].legend()
# plt.show()

# print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True))
# *Divide all values for this column to other columns and 1 means this row includes and 0 means excludes*
# print(pd.get_dummies(df['terms']))

