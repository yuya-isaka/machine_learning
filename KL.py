"""
ヒートマップで描画し可視化する
グラフを綺麗に表示できるらしいseabornというライブラリ使用
（各次元の値を出そうとしたら気持ち悪くなった）

https://qiita.com/sakuraya/items/bfcad038b9c2235c366f
"""

import pandas as pd

test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)

import seaborn as sns
import matplotlib.pyplot as plt

feature_names_x = ['%d' % i for i in range(test.shape[1])]
feature_names_y = ['%d' % i for i in range(test.shape[0])]

sns.heatmap(test, annot=False, xticklabels=feature_names_x, yticklabels=feature_names_y)

plt.show()
