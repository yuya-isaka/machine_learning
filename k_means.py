import numpy as np
import matplotlib.pyplot as plt

np.random.seed(69)
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
centers = np.array([[0, 5], [5, 0], [10, 15], [20, 10]])
idx = np.zeros(X.shape[0])

plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c="black", s=20, alpha=0.5)
plt.text(x=22, y=20, s="1", size=15)
plt.scatter(centers[:, 0], centers[:, 1], color=["r", "b", "g", "orange"])


#4回更新する
for j in np.arange(1, 4):
    #ラベル付
    for i in range(X.shape[0]):
        idx[i] = np.argmin(np.sum((X[i, :] - centers)**2, axis=1))
    #中心の更新
    for k in range(K):
        centers[k, :] = X[idx == k, :].mean(axis=0)

    plt.subplot(2, 2, j+1)
    plt.scatter(X[idx == 0, 0], X[idx == 0, 1], color="r", s=10, alpha=0.5)
    plt.scatter(X[idx == 1, 0], X[idx == 1, 1], color="b", s=10, alpha=0.5)
    plt.scatter(X[idx == 2, 0], X[idx == 2, 1], color="g", s=10, alpha=0.5)
    plt.scatter(X[idx == 3, 0], X[idx == 3, 1],
                color="orange", s=10, alpha=0.5)
    plt.text(x=22, y=20, s=str(j+1), size=15)
    plt.scatter(centers[:, 0], centers[:, 1], color=["r", "b", "g", "orange"])

plt.show()


def kmeans(X, K, centers, iter):
    idx = np.zeros(X.shape[0])
    for _ in range(iter):
        for i in range(X.shape[0]):
            idx[i] = np.argmin(np.sum((X[i, :]-centers)**2, axis=1))
        for k in range(K):
            centers[k,:] = X[idx==k,:].mean(axis=0)

    return idx



