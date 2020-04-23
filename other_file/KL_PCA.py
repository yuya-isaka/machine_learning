"""
主成分分析して相関行列を可視化しようとした
(対角成分が全部1になり，相関行列の特徴を満たしてるはず)

https://qiita.com/sakuraya/items/bfcad038b9c2235c366f
"""

import pandas as pd

test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)

import seaborn as sns
import matplotlib.pyplot as plt

feature_names_x = ['%d' % i for i in range(test.shape[1])]
feature_names_y = ['%d' % i for i in range(test.shape[0])]

from sklearn.decomposition import PCA
import numpy as np 

pca = PCA(n_components=30)
pca.fit(test)
features = pca.fit_transform(test)
matrix = np.corrcoef(features.transpose())

sns.heatmap(matrix, annot=True, xticklabels=feature_names_x, yticklabels=feature_names_y)

plt.show()