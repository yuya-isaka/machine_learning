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


"""確率密度関数の可視化"""
def plot_new(data):
    y = 11
    x = 5
    _, ax = plt.subplots(y,x,figsize=(30,30))
    count = 0
    for i in range(y):
        for j in range(x):
            if count >= 52:
                break
            dummy = sns.distplot(data[count], ax=ax[i, j])
            count += 1
        if count >= 52:
            break
    
    plt.show()

    
"""リストを繋げる関数"""
def connect(a, b):
    tmp = []
    for i in range(a.shape[0]):
        tmp.append(a[i])
        
    for i in range(b.shape[0]):
        tmp.append(b[i])
    
    tmp = np.array(tmp)
    return tmp


"""QQプロット出力する関数"""
def qq(x):
    stats.probplot(x, dist="norm", plot=plt)  #QQプロット表示
    plt.show()

    
"""コルモゴロフスミルノフ検定する関数"""
def kolmogorov_smirnov(x):
    loc, scale = norm.fit(x)
    n = norm(loc=loc, scale=scale)  # create a normal distribution with loc and scale
    p = stats.kstest(x, n.cdf).pvalue  #コルモゴロフスミルノフ検定
    return p


"""シャピロウィルク検定する関数"""
def shapiro_wilk(x):
    p=stats.shapiro(x)[1]  #シャピロウィルク検定
    return p


"""合計値と平均出す関数 一次元混合データが望ましい"""
def summean_residual(data):
    tmpsum = []
    tmpmean = []
    for i in range(data.shape[0]):
        tmpsum.append(data[i].sum())
        tmpmean.append(mean(data[i]))
        
    tmpsum = np.array(tmpsum)
    tmpmean = np.array(tmpmean)
    
    return tmpsum, tmpmean


"""散布図"""
def ten(data):
    x = np.arange(1, 53)
    plt.scatter(x, data)
    plt.title("scatter")
    plt.xlabel("sample number")
    plt.ylabel("residual")
    plt.grid(True)
    plt.show()

    
"""ホテリング理論"""
def hoteling(data):
    # 標本平均
    mn = mean(data)
    
    # 標本分散
    vari = variance(data)
    
    # 異常度
    anomaly_scores = []
    for x in data:
        anomaly_score = (x - mn)**2 / vari
        anomaly_scores.append(anomaly_score)
    
    # カイ二乗分布による5%水準の閾値
    threshold = stats.chi2.interval(0.95, 1)[1]
    
    # 結果の描画
    print('ホテリング理論結果')
    num = np.arange(1, 53)
    plt.plot(num, anomaly_scores, "o", color = "b")
    plt.plot([0,53],[threshold, threshold], 'k-', color = "r", ls = "dashed")
    plt.xlabel("Sample number")
    plt.ylabel("Anomaly score")
    plt.show()
    
    X = pd.DataFrame({"data": data})
    anomaly_scores_col = pd.DataFrame({"anomaly_score": anomaly_scores})
    X = pd.concat([X, anomaly_scores_col], axis=1)

    # 外れ値検知する
    normality = X[X["anomaly_score"] < threshold]
    outliers = X[threshold <= X["anomaly_score"]]

    # 正常値と外れ値をプロット
    print('散布図')
    plt.scatter(normality.index, normality["data"], label="normality")
    plt.scatter(outliers.index, outliers["data"], c="red", label="outlier")
    plt.ylabel("value")
    plt.legend()
    plt.show()
    
    for i, j in enumerate(anomaly_scores):
        if j >= threshold:
            print(f'異常index：{i+1}')
            
    print()

def kmeans_plus_plus(data, cluster_num):
    """
    中心点ゲット！
    """
    
    seeds = 50
    np.random.seed(seeds)

    # data_num = data.shape[0]
    feature_num = data.shape[0]

    centers = np.zeros(cluster_num)
    distance = np.zeros(cluster_num)

    probability = np.repeat(1/feature_num, feature_num)
    centers[0] = np.random.choice(data, 1, p=probability)
    distance[0] = np.sum((abs(data - centers[0]))**2)

    for k in range(1, cluster_num):
        np.random.seed(seeds*(k+1))
        probability = data / np.sum(distance)
        probability /= probability.sum() #正規化　probabilities do not sum to 1　と怒られたから

        centers[k] = np.random.choice(data, 1, p=probability)
        distance[k] = np.sum((abs(data - centers[k]))**2)

    return centers


import matplotlib.pyplot as plt


def wrap_k_means(data, k=3):
    cluster = kmeans_plus_plus(data, k) # 初期値kmeans++
    prof, cluster = k_means(data, cluster)
    return prof, cluster

def k_means(data, cluster):
    data = np.array(data, dtype=float)
    prof = np.zeros(len(data)) # 各要素∈dataがどのクラスタに属しているか
    cluster = np.sort(np.array(cluster, dtype=float))
    old_cluster = np.zeros(len(cluster)) # 収束チェック用
    
    conv = True; count = 0
    while conv:
        count += 1
        
        # 割り当て
        for i,d in enumerate(data):
            min_d = float('inf')# てきとうに大きい数
            for j,c in enumerate(cluster):
                dist = (abs(d - c))**2
                if min_d > dist:
                    min_d = dist
                    prof[i] = j # クラスタの割り当て

        # 更新
        for j,c in enumerate(cluster):
            m = 0; n = 0
            for i,p in enumerate(prof):
                if p == j: # もしもそのクラスタに属していたら
                    m += data[i]
                    n += 1
            if m != 0:
                m /= n # mは更新した平均
                old_cluster[j] = cluster[j]
                cluster[j] = m
            
        # 途中経過
        # print("{}回目".format(count))
        # print("data   : ", data)
        # print("prof   : ", prof)
        # print("cluster: ", cluster)
        
        # 収束チェック
        for i,c in enumerate(cluster):
            if c != old_cluster[i]:
                conv = True
                break
            else:
                conv = False

    # 結果出力
    # print("result")
    # print("data   : ", data) # debug
    # print("prof   : ", prof) # debug
    # print("cluster: ", cluster) # debug    

    return prof, cluster

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
    freshaged = connect(fresh, aged) # (52, 148, 33)
    flat_freshaged = change_flatten(freshaged) # (52, 4884)
    nonzero_freshaged = delete_zero(flat_freshaged) #(52, 3964前後)

    every_cluster_list = [] # (7, 52)
    for x in range(2, 9):

        fpga_class = []
        for i in range(nonzero_freshaged.shape[0]):
            prof, cluster = wrap_k_means(nonzero_freshaged[i], x)
            print(f'{i}回目')
            tmp = FPGA(nonzero_freshaged[i], prof, cluster)
            fpga_class.append(tmp)

        # frequency, cluster, center
        x_cluster_list = [] #(52)
        counter = 0
        for i in fpga_class:
            num = len(i.center)
            sci_mean_list = [] #(num)
            for j in range(num):

                tmp_list = [] 
                for k in range(len(i.frequency)):
                    if i.cluster[k] == j:
                        tmp_list.append(i.frequency[k])
                tmp_list = np.array(tmp_list)

                tmp_list2 = choose_center(i, j, tmp_list)

                #シルエットプロット計算
                sci_list = [] 
                for k in tmp_list:
                    oci = np.sum(abs(tmp_list - k)) / (len(tmp_list) - 1)
                    nci = np.sum(abs(tmp_list2 - k)) / len(tmp_list2)
                    sci = (nci - oci) / max(nci, oci)
                    sci_list.append(sci)

                sci_list = np.array(sci_list)
                sci_mean = np.mean(sci_list)
                sci_mean_list.append(sci_mean)

            sci_mean_list = np.array(sci_mean_list)
            sci_mean_mean_list = np.mean(sci_mean_list)
            x_cluster_list.append(sci_mean_mean_list)
            print(f'{counter}回目')
            counter += 1

        every_cluster_list.append(x_cluster_list)
        print(f'{x}回目')

    f = open('ACN_list_3.binaryfile', 'wb')
    pickle.dump(every_cluster_list, f)
    f.close()


if __name__ == "__main__":
    main()
    pass