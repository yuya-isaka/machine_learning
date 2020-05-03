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

"""東西南北残差作り出す関数"""
def EWSN_residual(data):
    tmp_x = [0, 1, 0, -1]
    tmp_y = [-1, 0, 1, 0]

    residual_data = np.zeros_like(data)

    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            for k in range(data[i].shape[1]):
                if data[i, j, k] != 0:
                    data_list = []
                    for l in range(4):
                        next_y = j + tmp_y[l]
                        next_x = k + tmp_x[l]
                        if 0 <= next_y < 148 and 0 <= next_x < 33 and data[i, next_y, next_x] != 0:
                            data_list.append(data[i, next_y, next_x])
    
                    data_mean = mean(data_list)
                    residual_data[i, j, k] = abs(data[i, j, k] - data_mean)

    return residual_data

"""king残差作り出す関数"""
def king_residual(data):
    tmp_x = [-1, 0, 1, 1, 1, 0, -1, -1]
    tmp_y = [-1, -1, -1, 0, 1, 1, 1, 0]
    
    residual_data = np.zeros_like(data)
    
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            for k in range(data[i].shape[1]):
                if data[i, j, k] != 0:
                    data_list = []
                    for l in range(8):
                        next_y = j + tmp_y[l]
                        next_x = k + tmp_x[l]
                        if 0 <= next_y < 148 and 0 <= next_x < 33 and data[i, next_y, next_x] != 0:
                            data_list.append(data[i, next_y, next_x])
    
                    data_mean = mean(data_list)
                    residual_data[i, j, k] = abs(data[i, j, k] - data_mean)

    return residual_data

"""確率密度関数の可視化"""
def plot_new(data):
    y = 11
    x = 5
    fig, ax = plt.subplots(y,x,figsize=(30,30))
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
    plt.pause(.01)

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
    num = np.arange(1, 53)
    plt.plot(num, anomaly_scores, "o", color = "b")
    plt.plot([0,53],[threshold, threshold], 'k-', color = "r", ls = "dashed")
    plt.xlabel("Sample number")
    plt.ylabel("Anomaly score")
    plt.show()
    
    for i, j in enumerate(anomaly_scores):
        if j >= threshold:
            print(f'異常index：{i+1}')






