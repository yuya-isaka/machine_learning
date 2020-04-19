import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  

np.random.seed(2020)

plt.style.use('default')
sns.set()
sns.set_style('whitegrid')

# make 10x8 matrix
# data = np.random.binomial(100, 0.02, 80).reshape((10, 8))
# data = np.log2(data + 1)

df = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)

# df = pd.DataFrame(data,
#                   # リスト内包表記！！！！   strで繋ぐのが18番ですね。
#                   index=['gene ' + str(i + 1) for i in range(10)],
#                   columns=['sample ' + str(i + 1) for i in range(8)])

print(df)


# heatmap
sns_plot = sns.clustermap(df, method='ward', metric='euclidean')
plt.setp(sns_plot.ax_heatmap.get_yticklabels(), rotation=0)
plt.setp(sns_plot.ax_heatmap.get_xticklabels(), rotation=30)

plt.show()