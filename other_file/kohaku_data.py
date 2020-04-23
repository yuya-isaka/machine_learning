# サンプルデータ作成
import numpy as np

# seed値固定
np.random.seed(874)
# x座標
x = np.r_[np.random.normal(size=1000,loc=0,scale=1),
          np.random.normal(size=1000,loc=4,scale=1)]
# y座標
y = np.r_[np.random.normal(size=1000,loc=10,scale=1),
          np.random.normal(size=1000,loc=10,scale=1)]
data = np.c_[x, y]

print(x)
print(x.shape)
print(y)
print(y.shape)

print(data)
print(data.shape)

print(data[:,0])
print(data[:,1])



# 可視化処理
# import matplotlib.pyplot as plt

# p = plt.subplot()
# p.scatter(data[:,0], data[:,1], c = "black", alpha = 0.5)
# p.set_aspect('equal')
# plt.show()