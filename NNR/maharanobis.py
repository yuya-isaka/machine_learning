import numpy as np
import pandas as pd
from sklearn import datasets
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from statistics import mean, variance

def csv_to_data(directory, data_n):
    data = []
    
    for i in range(1, data_n+1):
        tmp_data = pd.read_csv(directory+'/s'+str(i)+'.csv', header=None).values
        data.append(tmp_data)
    
    return np.array(data)


def csv_to_aged_data(directory, aged_data_n):
    aged_data = []

    for i in range(1, aged_data_n+1):
        tmp_data = pd.read_csv(directory+'/s'+str(i)+'_aged.csv', header=None).values
        aged_data.append(tmp_data)
        
    return np.array(aged_data)


def generate_data(directory, data_n, aged_data_n):
    data = csv_to_data(directory, data_n)
    aged_data = csv_to_aged_data(directory, aged_data_n)
    
    return data, aged_data


def generate_residual_data(data_n, data):
    """
    測定値-推定値(周りの平均)
    残差を求めてデータ生成
    """
    tmp_x = [0, 1, 0, -1]
    tmp_y = [-1, 0, 1, 0]

    residual_data = np.zeros_like(data)

    for i in range(data_n):
        for j in range(data[i].shape[0]):
            for k in range(data[i].shape[1]):
                data_list = []
                for l in range(4):
                    next_y = j + tmp_y[l]
                    next_x = k + tmp_x[l]
                    if 0 <= next_y < 148 and 0 <= next_x < 33:
                        data_list.append(data[i, next_y, next_x])

                data_mean = mean(data_list)
                residual_data[i, j, k] = np.abs(data[i, j, k] - data_mean)

    return residual_data


def generate_nnr(data_n=50, aged_data_n=2):
    """
    残差集合のデータ生成
    """
    data, aged_data = generate_data('fresh_aged_ieice', data_n, aged_data_n)

    residual_data = generate_residual_data(data_n, data)
    aged_residual_data = generate_residual_data(aged_data_n, aged_data)

    return residual_data, aged_residual_data

a, b = generate_data('fresh_aged_ieice', 50, 2)

data = []
for i in range(50):
    data.append(a[i].flatten())
    
for i in range(2):
    data.append(b[i].flatten())
    
data = np.array(data)

# 正常データの設定
#iris = datasets.load_iris()

df_data = pd.DataFrame(data, columns=np.arange(1,4885))

# 異常データの設定

from sklearn.model_selection import train_test_split

# データを7:3で学習用とテスト用に分割
df_data_train, df_data_test = train_test_split(df_data, test_size=0.3, random_state=3655)

df_x_train = df_data_train
df_x_test = df_data_test

from sklearn.covariance import EmpiricalCovariance, MinCovDet

# MCD
mcd = MinCovDet()
mcd.fit(df_x_train)
anomaly_score_mcd = mcd.mahalanobis(df_x_test)

# 最尤法
mle = EmpiricalCovariance()
mle.fit(df_x_train)
anomaly_score_mle = mle.mahalanobis(df_x_test)
