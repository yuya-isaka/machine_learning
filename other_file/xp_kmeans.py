import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')

def generate_data(directory, n, aged_n):
    """与えられたcsvファイルからデータを取り出し，(データ数，特徴量の形に前処理する)
    Arguments:
        directory [str]     -- [フォルダ名]
        n [str]             -- [正常FPGAのCSVデータ数]
        aged_n [str]        -- [経年劣化FPGAのCSVデータ数]
    
    Returns:
        [sequence] -- [正常FPGAのCSVデータ配列]
        [sequence] -- [経年劣化FPGAのCSVデータ配列]
    """
    data = []
    aged_data = []
    for i in range(1, n+1):
        tmp = pd.read_csv(directory+'/s'+str(i)+'.csv', header=None).values
        tmp2 = []
        for j in range(tmp.shape[0]): # 148
            for k in range(tmp.shape[1]): # 33
                tmp2.append([j, k, tmp[j][k]])
        data.append(tmp2)
        if 1 <= i <= aged_n:
            tmp = pd.read_csv(directory+'/s'+str(i)+'_aged.csv', header=None).values
            tmp2 = []
            for j in range(tmp.shape[0]):
                for k in range(tmp.shape[1]):
                    tmp2.append([j, k, tmp[j][k]])
            aged_data.append(tmp2)

    return np.array(data), np.array(aged_data)


def kmeans(X, K, centers, iterate):
    """kmeansは中心点とその周りのクラスタの値を変更することを繰り返す関数
    
    Arguments:
        X {[int]} -- [二次元配列(x座標は特徴量, y座標はデータ数)]
        K {[int]} -- [クラスタ数]
        centers {[sequence]} -- [指定したクラスタ数の中心点]
        iterate {[int]} -- [kmeansの中心点を変更してクラスタリングする回数]
    
    Returns:
        idx [sequence]      -- [各場所のクラスタされたインデックス]
        sse_sum [sequence]  -- [クラスタ内誤差平方和，それぞれのクラスタの，中心とそれぞれの点との差の二乗をした和の，全てのクラスタの和]
    """
    idx = np.zeros(X.shape[0])
    sse = np.zeros(K)
    for _ in range(iterate):

        for i in range(X.shape[0]):
            idx[i] = np.argmin(np.sum((X[i, :]-centers)**2, axis=1))

        for k in range(K):
            centers[k, :] = X[idx == k, :].mean(axis=0)
            sse[k] = np.sum((X[idx == k, :]-centers[k, :])**2)

    sse_sum = np.sum(sse)

    return idx, sse_sum, centers



def kmeans_plus_plus(X, K, iterate):
    """kmeans_plus_plusは，最初の中心点をばらつくように決定し，kmeansアルゴリズムに最初の中心点を渡し，クラスタリングの値とSSEを返してくれる関数

    Arguments:
        X [int]             -- [二次元配列 (x座標は特徴量 y座標はデータ数)]
        K [int]             -- [クラスタ数]
        iterate [int]       -- [kmeansの中心点を変更してクラスタリングする回数]
    
    Returns:
        idx [sequence]      -- [各場所のクラスタされたインデックス]
        sse_sum [sequence]  -- [クラスタ内誤差平方和，それぞれのクラスタの，中心とそれぞれの点との差の二乗をした和の，全てのクラスタの和]
    """
    n = X.shape[0]  # 4884
    seeds = 39  # seed値適当に
    np.random.seed(seeds)
    tmp = np.arange(n)
    distance = np.zeros(n*K).reshape(n, K)  # (4884 K) それぞれの場所の
    # -1で上手いこと合わせてくれる (K, 3) # 各クラスタ中心点の座標を格納
    centers = np.zeros(X.shape[1]*K).reshape(K, -1)

    pr = np.repeat(1/n, n)
    centers[0, :] = X[np.random.choice(tmp, 1, p=pr)]
    distance[:, 0] = np.sum((X-centers[0, :])**2, axis=1)

    for k in range(1, K):
        np.random.seed(seeds*(k+1))
        pr = np.sum(distance, axis=1) / np.sum(distance)
        centers[k, :] = X[np.random.choice(tmp, 1, p=pr)]
        distance[:, k] = np.sum((X-centers[k, :])**2, axis=1)

    idx, sse_sum, new_centers = kmeans(X, K, centers, iterate)

    return idx, sse_sum, new_centers

def renovate(idx, csv_data):
    """
    概要：クラスタ番号が与えられた配列とcsvデータを与えたら，復元したデータを返す

    受け取り：idx((1, 4884)の一意のクラスタ番号を与えられたFPGA一つの配列)
              csv_data(対応するFPGAの(`148, 33)のcsvデータ)

    返す：new((4884, 4)の復元したデータにクラスタ番号を加えたもの)
    """
    count = 0
    new = []
    for i in range(csv_data.shape[0]):
        for j in range(csv_data.shape[1]):
            tmp = [i, j, csv_data[i][j], idx[count]]
            new.append(tmp)
            count += 1

    return np.array(new)

def center_neighbor(centers, search_num):
    """
    ・中心点が入った配列と調べたい中心点のインデックスを渡す．
    ・1番近いインデックスを返してくれる
    """
    ans = 0
    min_value = 1e9
    for i in range(centers.shape[0]):
        if(i == search_num):
            continue
        tmp = np.sum((centers[search_num, :] - centers[i, :])**2, axis=1)
        if min_value > tmp:
            ans = i
            min_value = tmp

    return ans


def silhouette(centers, cluster_num, neighbor_num, renovate_data):
    """
    ・中心点の特徴量，調べたいクラスタ番号，調べたいクラスタ番号の近傍のクラスタ番号，復元したデータを渡す
    ・sviを返す
    """
    delete_data = np.delete(renovate_data, 3, 1)
    cluster_self = 0
    cluster_neighbor = 0
    for i in range(renovate_data.shape[0]):
        if renovate_data[i][3] == cluster_num:
            cluster_self += np.sum((centers[cluster_num, :] - delete_data[i, :])**2, axis=1)
        if renovate_data[i][3] == neighbor_num:
            cluster_neighbor += np.sum((centers[neighbor_num, :] - delete_data[i, :])**2, axis=1)

    svi = cluster_self - cluster_neighbor
    return svi


new, used = generate_data('fresh_aged_ieice', 50, 2)
print(new.shape)
print('-'*100)
print(used.shape)

iterator = 10
idx_list = []
sse_list = []
idx_list_aged = []
sse_list_aged = []
for k in range(2, 3):
    for i in range(50):
        tmp, tmp_sse = kmeans_plus_plus(new[i], k, iterator)
        idx_list.append(tmp)
        sse_list.append(tmp_sse)
        if 0 <= i <= 1:
            tmp2, tmp2_sse = kmeans_plus_plus(used[i], k, iterator)
            idx_list_aged.append(tmp2)
            sse_list_aged.append(tmp2_sse)
        print('{}回目調べて終えました'.format(i+1))
        print('結果{}'.format(np.array(idx_list).shape))

idx_list = np.array(idx_list)
sse_list= np.array(sse_list)
idx_list_aged = np.array(idx_list_aged)
sse_list_aged = np.array(sse_list_aged)

print(idx_list.shape)
print(sse_list_aged.shape)

# K = 10
# iterator = 10
# sse_vec = np.zeros(K)
# for k in range(K):
#     idx, sse = kmeansplus(X,k+1,iterator)
#     sse_vec[k] = sse

# plt.plot(np.arange(1,11),sse_vec)
# plt.plot(np.arange(1,11),sse_vec,"bo")
# plt.show()

