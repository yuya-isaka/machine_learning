
"""ライブラリ"""
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from statistics import mean, variance
from scipy import stats
from scipy.stats import norm
from collections import Counter
import copy
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import cm

class FPGA:
    def __init__(self,frequency, cluster, center):
        self.frequency = frequency
        self.cluster = cluster
        self.center = center

"""データ生成関数"""
def generate_data(directory, data_n, aged_data_n):
    data = []
    aged_data = []
    
    for i in range(1, data_n+1):
        tmp_data = pd.read_csv(directory+'/s'+str(i)+'.csv', header=None).values
        data.append(tmp_data)
    
    for i in range(1, aged_data_n+1):
        tmp_data = pd.read_csv(directory+'/s'+str(i)+'_aged.csv', header=None).values
        aged_data.append(tmp_data)
    
    data = np.array(data)
    aged_data = np.array(aged_data)
    
    return data, aged_data

newdata, ageddata = generate_data('fresh_aged_ieice', 50, 2)


"""0の数数える関数 二次元入れろ"""
def count_zero(data):
    
    tmp = []
    for i in range(data.shape[0]):
        a = (data[i].shape[0] * data[i].shape[1]) - np.count_nonzero(data[i])
        tmp.append(a)
        
    return tmp

    """９２０この値を消す"""
def delete_920(data, check):
    new = np.zeros_like(data)
    counter = 0
    for i in data:
        for j in range(i.shape[0]):
            for k in range(i.shape[1]):
                if [j, k] in check:
                    new[counter, j, k] = 0 
                else:
                    new[counter, j, k] = data[counter, j, k]
                    
        counter += 1
                    
    return new


"""一次元にする関数"""
def change_flatten(data):
    tmp = []
    for i in range(data.shape[0]):
        tmp.append(data[i].flatten())
        
    tmp = np.array(tmp)
    
    return tmp


"""0を消す関数 flatteしたやつ入れろ"""
def delete_zero(data):
    tmp = []
    for i in range(data.shape[0]):
        tmp2 = copy.deepcopy(data[i])
        tmp.append(tmp2[tmp2 != 0])
        
    tmp = np.array(tmp)
    
    return tmp


"""リストを繋げる関数"""
def connect(a, b):
    tmp = []
    for i in range(a.shape[0]):
        tmp.append(a[i])
        
    for i in range(b.shape[0]):
        tmp.append(b[i])
    
    tmp = np.array(tmp)
    return tmp

def choose_center(i, x, tmp_list):
    tmp = []
    for j in range(len(i.center)):
        if j != x:
            tmp_list2 = []
            for k in range(len(i.frequency)):
                if i.cluster[k] == j:
                    tmp_list2.append(i.frequency[k])
            tmp.append(tmp_list2)

    tmp = np.array(tmp)

    nci_list = []
    for k in tmp:
        average_list = []
        for j in tmp_list:
            nci = np.sum(abs(k - j)) / len(k)
            average_list.append(nci)

        average_list = np.array(average_list)
        average_mean = np.mean(average_list)

        nci_list.append(average_mean)

    nci_list = np.array(nci_list)

    a = np.argmin(nci_list)

    return tmp[a]


def main():
    fresh, aged = generate_data('fresh_aged_ieice', 50, 2) # (50, 148, 33) (2, 148, 33)

    newdata = connect(fresh, aged) # (52, 148, 33)
    check_FPGA = []
    for i in range(148):
        for j in range(33):
            if fresh[0, i, j] == 0:
                check_FPGA.append([i,j])
    freshaged = delete_920(newdata, check_FPGA) #(52, 148, 33)

    new_freshaged = change_flatten(freshaged) # (52 4884)

    nonzero_freshaged = delete_zero(new_freshaged) #(52, 3964)

    new = []
    for i in nonzero_freshaged:
        tmp = []
        for j in range(len(i)):
            tmp.append([i[j], 0])

        new.append(tmp)

    new = np.array(new)

    ans = []
    for i in range(2, 9):
        tmp = []
        for j in range(new.shape[0]):
            km = KMeans(n_clusters=i,            # クラスターの個数
                init='k-means++',        # セントロイドの初期値をランダムに設定
                n_init=10,               # 異なるセントロイドの初期値を用いたk-meansあるゴリmズムの実行回数
                max_iter=300,            # k-meansアルゴリズムの内部の最大イテレーション回数
                tol=1e-04,               # 収束と判定するための相対的な許容誤差
                random_state=0)          # セントロイドの初期化に用いる乱数発生器の状態
            y_km = km.fit_predict(new[j])

            # cluster_labels = km.labels_

            # cluster_labels = np.unique(y_km)       # y_kmの要素の中で重複を無くす
            # n_clusters=cluster_labels.shape[0]     # 配列の長さを返す。つまりここでは n_clustersで指定した3となる
            # シルエット係数を計算
            #silhouette_vals = silhouette_samples(new[j],y_km,metric='euclidean')  # サンプルデータ, クラスター番号、ユークリッド距離でシルエット係数計算
            # y_ax_lower, y_ax_upper= 0,0
            # yticks = []

            # for n_clusters in range_n_clusters:
            #     clusterer = KMeans(n_clusters=n_clusters)
            #     preds = clusterer.fit_predict(df)
            #     centers = km.cluster_centers_

            score = silhouette_score(new[j], y_km)
                # print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
            # for i,c in enumerate(cluster_labels):
            #         c_silhouette_vals = silhouette_vals[y_km==c]      # cluster_labelsには 0,1,2が入っている（enumerateなのでiにも0,1,2が入ってる（たまたま））
            #         c_silhouette_vals.sort()
            #         y_ax_upper += len(c_silhouette_vals)              # サンプルの個数をクラスターごとに足し上げてy軸の最大値を決定
            #         color = cm.jet(float(i)/n_clusters)               # 色の値を作る
            #         plt.barh(range(y_ax_lower,y_ax_upper),            # 水平の棒グラフのを描画（底辺の範囲を指定）
            #                          c_silhouette_vals,               # 棒の幅（1サンプルを表す）
            #                          height=1.0,                      # 棒の高さ
            #                          edgecolor='none',                # 棒の端の色
            #                          color=color)                     # 棒の色
            #         yticks.append((y_ax_lower+y_ax_upper)/2)          # クラスタラベルの表示位置を追加
            #         y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加

            #silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
            # plt.axvline(silhouette_avg,color="red",linestyle="--")    # 係数の平均値に破線を引く 
            # plt.yticks(yticks,cluster_labels + 1)                     # クラスタレベルを表示
            # plt.ylabel('Cluster')
            # plt.xlabel('silhouette coefficient')
            # plt.show()
            # kmeans_model = KMeans(n_clusters=x, init='k-means++').fit(new[i])
            # cluster = kmeans_model.cluster_centers_[0]

            tmp.append(score)

            print(f'{j}回目')

        ans.append(tmp)

    f = open('ACN_list_7.binaryfile', 'wb')
    pickle.dump(ans, f)
    f.close()


if __name__ == "__main__":
    main()
    pass


