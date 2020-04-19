import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans


"""指紋を地形のように見立てて表現
"""
# test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)
# test2 = pd.read_csv('fresh_aged_ieice/s1_aged.csv', header=None)
# frequency = test.values
# frequency2 = test2.values

# x = np.empty([148, 33])
# y = np.empty([148, 33])

# for i in range(148):
#     for j in range(33):
#         x[i,j] = j
#         y[i,j] = i

# print(x)
# print(y)

# plt.subplot(2,1,1)
# plt.contourf(x,y,frequency)
# plt.colorbar()

# plt.subplot(2,1,2)
# plt.contourf(x,y,frequency2)
# plt.colorbar()
# plt.show()

"""seabornを使ったheatmap表現
"""
# feature_names_x = ['%d' % i for i in range(test.shape[1])]
# feature_names_y = ['%d' % i for i in range(test.shape[0])]

# sns.heatmap(test, annot=False, xticklabels=feature_names_x, yticklabels=feature_names_y)

# plt.show()


"""sklearnのKMeansを使ったバージョン
"""
# test = []
# for i in range(1,50):
#     tmp = pd.read_csv('fresh_aged_ieice/s'+str(i)+'.csv', header=None)
#     test.append(tmp.values)

# pred = KMeans(n_clusters=2).fit_predict(test)
# print(pred)


"""kmeans関数とkmeans++関数
"""
# def kmeans(X, K, centers, iterate):
#     idx = np.zeros(X.shape[0])
#     for _ in range(iterate):
#         for i in range(X.shape[0]):
#             idx[i] = np.argmin(np.sum((X[i, :]-centers)**2, axis=1))
#         for k in range(K):
#             centers[k,:] = X[idx==k,:].mean(axis=0)

#     return idx

# def kmeansplus(X, K,iterate):
#     n = X.shape[0]
#     distance = np.zeros(n*K).reshape(n, K)
#     centers = np.zeros(X.shape[1]*K).reshape(K, -1)

#     pr = np.repeat(1/n, n)
#     centers[0, :] = X[np.random.choice(np.arange(n), 1, p=pr)]
#     distance[:, 0] = np.sum((X-centers[0, :])**2, axis=1)

#     for k in np.arange(1, K):
#         pr = np.sum(distance, axis=1)/np.sum(distance)
#         centers[k, :] = X[np.random.choice(np.arange(n), 1, p=pr)]
#         distance[:, k] = np.sum((X-centers[k, :])**2, axis=1)

#     idx = kmeans(X,K,centers,iterate)

#     return idx

# x1 = np.r_[np.random.normal(size=20, loc=1, scale=2),
#            np.random.normal(size=20, loc=8, scale=2),
#            np.random.normal(size=20, loc=15, scale=2),
#            np.random.normal(size=20, loc=25, scale=2)]
# x2 = np.r_[np.random.normal(size=20, loc=15, scale=2),
#            np.random.normal(size=20, loc=1, scale=2),
#            np.random.normal(size=20, loc=20, scale=2),
#            np.random.normal(size=20, loc=0, scale=2)]
# X = np.c_[x1, x2]  # (80, 2)

# plt.scatter(X[:, 0], X[:, 1], c="black", s=10, alpha=0.5)
# plt.scatter(centers[:, 0], centers[:, 1], color=["r", "b", "g", "orange"])
# plt.show()


"""xとyとzを用意
"""
# test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)
# test_value = np.array(test.values)

# X = np.array([i for i in range(0,33)]) #自然数の配列
# Y = np.array([i for i in range(0,148)]) #特に意味のない正弦
# x = np.arange(0,33,1)
# y = np.arange(0,148,1)

# z = test_value[y][x]


"""kmeansとkmeans++ sse算出版
"""
# def kmeans(X, K, centers, iterate):
#     idx = np.zeros(X.shape[0])
#     #クラスごとのsseを格納するための入れ物
#     sse = np.zeros(K)
#     for _ in range(iterate):

#         for i in range(X.shape[0]):
#             idx[i] = np.argmin(np.sum((X[i, :]-centers)**2, axis=1))

#         for k in range(K):
#             centers[k, :] = X[idx == k, :].mean(axis=0)
#             sse[k] = np.sum((X[idx==k,:]-centers[k,:])**2)

#     sse_sum = np.sum(sse)

#     return idx, sse_sum

# #kmeansとkmeans++の違いは，最初の中心点の座標をランダムに決めるか決めないか．それだけ．
# def kmeansplus(X, K, iterate):
#     n = X.shape[0]
#     distance = np.zeros(n*K).reshape(n, K)
#     centers = np.zeros(X.shape[1]*K).reshape(K, -1)

#     pr = np.repeat(1/n, n)
#     centers[0, :] = X[np.random.choice(np.arange(n), 1, p=pr)]
#     distance[:, 0] = np.sum((X-centers[0, :])**2, axis=1)

#     for k in np.arange(1, K):
#         pr = np.sum(distance, axis=1)/np.sum(distance)
#         centers[k, :] = X[np.random.choice(np.arange(n), 1, p=pr)]
#         distance[:, k] = np.sum((X-centers[k, :])**2, axis=1)

#     idx, sse_sum = kmeans(X, K, centers, iterate)

#     return idx, sse_sum


# K = 10
# iterator = 10
# sse_vec = np.zeros(K)
# for k in range(K):
#     idx, sse = kmeansplus(X,k+1,iterator)
#     sse_vec[k] = sse

# plt.plot(np.arange(1,11),sse_vec)
# plt.plot(np.arange(1,11),sse_vec,"bo")
# plt.show()