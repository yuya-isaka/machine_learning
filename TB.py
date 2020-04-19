# 基本のライブラリを読み込む
import numpy as np
import pandas as pd
from scipy import stats

# グラフ描画
from matplotlib import pylab as plt
import seaborn as sns

# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

import matplotlib as mpl
mpl.rcParams['font.family'] = ['serif']

url = 'https://docs.google.com/spreadsheets/d/1X5Jp7Q8pTs3KLJ5JBWKhncVACGsg5v4xu6badNs4C7I/pub?gid=0&output=csv'
existing_df = pd.read_csv(url,
    index_col = 0, 
    thousands  = ',')
existing_df.index.names = ['country']
existing_df.columns.names = ['year']
dhead = existing_df.head()
#print(dhead)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(existing_df)

# print("--- explained_variance_ratio_ ---")
# print(pca.explained_variance_ratio_)
# print("--- components ---")
# print(pca.components_)
# print(pca.components_.shape)
# print("--- mean ---")
# print(pca.mean_)
# print("--- covariance ---")
# print(pca.get_covariance())

existing_2d = pca.transform(existing_df)
existing_df_2d = pd.DataFrame(existing_2d)
existing_df_2d.index = existing_df.index
existing_df_2d.columns = ['PC1','PC2']
dhead = existing_df_2d.head()
#print(dhead)

ax = existing_df_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8))
for i, country in enumerate(existing_df.index):
    ax.annotate(  
        country,
       (existing_df_2d.iloc[i].PC2, existing_df_2d.iloc[i].PC1)
    )