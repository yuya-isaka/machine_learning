import numpy as np #適当な配列作るためにNumpy使う
import pandas as pd

test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)
test_value = np.array(test.values)



X = np.array([i for i in range(0,33)]) #自然数の配列
Y = np.array([i for i in range(0,148)]) #特に意味のない正弦
x = np.arange(0,33,1)
y = np.arange(0,148,1)
frequency = test_value[y][x] #特に意味のない自然対数
#備考：Numpyだとnp.log()は自然対数。常用対数はnp.log10()

import matplotlib.pyplot as plt

#seabornでグラフをきれいにしたいだけのコード
import seaborn as sns
sns.set_style("darkgrid")

#3次元プロットするためのモジュール
from mpl_toolkits.mplot3d import Axes3D

#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)


#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("frequency")

#.plotで描画
#linestyle='None'にしないと初期値では線が引かれるが、3次元の散布図だと大抵ジャマになる
#markerは無難に丸
ax.plot(X,Y,frequency,marker="o",linestyle='None')

#最後に.show()を書いてグラフ表示
plt.show()