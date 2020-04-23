import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

def kmeans(X, K, centers, iterate):
    idx = np.zeros(X.shape[0])
    for _ in range(iterate):
        #各点を,中心情報を元にラベル付する
        for i in range(X.shape[0]):
            idx[i] = np.argmin(np.sum((X[i, :]-centers)**2, axis=1))
        #平均の値を求める
        for k in range(K):
            centers[k, :] = X[idx == k, :].mean(axis=0)

    return idx

#kmeansとkmeans++の違いは，最初の中心点の座標をランダムに決めるか決めないか．それだけ．
def kmeansplus(X, K, iterate):
    n = X.shape[0]
    distance = np.zeros(n*K).reshape(n, K)
    centers = np.zeros(X.shape[1]*K).reshape(K, -1)

    #最初の確率は均等
    pr = np.repeat(1/n, n)
    #1つ目の中心点はランダムに選ぶ
    centers[0, :] = X[np.random.choice(np.arange(n), 1, p=pr)]
    distance[:, 0] = np.sum((X-centers[0, :])**2, axis=1)

    #決めてあったクラスターの数繰り返す
    for k in np.arange(1, K):
        pr = np.sum(distance, axis=1)/np.sum(distance)
        centers[k, :] = X[np.random.choice(np.arange(n), 1, p=pr)]
        distance[:, k] = np.sum((X-centers[k, :])**2, axis=1)

    idx = kmeans(X, K, centers, iterate)

    return idx


x1 = np.r_[np.random.normal(size=20, loc=1, scale=2),
           np.random.normal(size=20, loc=8, scale=2),
           np.random.normal(size=20, loc=15, scale=2),
           np.random.normal(size=20, loc=25, scale=2)]
x2 = np.r_[np.random.normal(size=20, loc=15, scale=2),
           np.random.normal(size=20, loc=1, scale=2),
           np.random.normal(size=20, loc=20, scale=2),
           np.random.normal(size=20, loc=0, scale=2)]
X = np.c_[x1, x2]  # (80, 2)

#初期値
K = 4
iterate = 10

idx = kmeansplus(X, K, iterate)

plt.scatter(x1[idx == 0], x2[idx == 0], color="r", s=10, alpha=0.5)
plt.scatter(x1[idx == 1], x2[idx == 1], color="b", s=10, alpha=0.5)
plt.scatter(x1[idx == 2], x2[idx == 2], color="g", s=10, alpha=0.5)
plt.scatter(x1[idx == 3], x2[idx == 3], color="orange", s=10, alpha=0.5)
plt.show()
