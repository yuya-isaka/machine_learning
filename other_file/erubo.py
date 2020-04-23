"""
エルボー法という，タスクに最適なクラスタの個数kを推定できる．
クラスタ内誤差平方和（歪み）が最も急速に増え始めるkの値を特定する．

エルボーが3にあるからk=3がいいということになる．らしい．
ギャグかよ
"""
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1],c='white', marker='o', edgecolors='black', s=50)
plt.grid()
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
dis = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    dis.append(km.inertia_)

plt.plot(range(1,11), dis, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Dis')
plt.tight_layout()
plt.show()


