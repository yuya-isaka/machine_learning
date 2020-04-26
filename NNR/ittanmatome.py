import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from statistics import mean, variance
from scipy import stats

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

# a, b = generate_data('fresh_aged_ieice', 50, 2)

# data = []
# for i in range(50):
#     data.append(a[i].flatten())
    
# for i in range(2):
#     data.append(b[i].flatten())
    
# data = np.array(data)

# y = 11
# x = 5
# fig,ax = plt.subplots(y,x,figsize=(20,20))
# count = 0
# for i in range(y):
#     for j in range(x):
#         if count >= 52:
#             break
#         dummy = sns.distplot(data[count], ax=ax[i, j])
#         count += 1
#     if count >= 52:
#         break

# plt.show()

"""
異常検知にはいろいろアプローチがある
距離に基づくアプローチ
統計学に基づくアプローチ
機械学習に基づくアプローチ

まず，統計学に基づくアプローチを行う．
これは統計的異常検知と呼ばれ，データがある確率分布モデルから生成されていると仮定した場合の異常検知の方法論です．
データから得られた情報を確率モデルという形で表現していると取れる．

3ステップからなる
1.観測データからデータ生成の確率モデルを生成する
　さらに細かく
未知パラメータを含む確率分布を仮定
データから未知パラメータを推定
2. 学習したモデルを基に，データの異常度あいをスコアリング
3. 闘値の決定

今回扱うデータは50個のFPGAの残差データの合計です．（まずは一次元でやる，その後に多次元に拡張する）何か得られるデータがあるかもしれない

まずは可視化してみよう．
"""

# residual_data, aged_residual_data = generate_nnr()

# data = []
# for i in range(50):
#     data.append(residual_data[i].sum())
    
# for i in range(2):
#     data.append(aged_residual_data[i].sum())
    
# data = np.array(data)

# x = np.arange(1, 53)
# plt.scatter(x, data)
# plt.title("scatter")
# plt.xlabel("sample number")
# plt.ylabel("residual")
# plt.grid(True)
# plt.show()

"""
横軸を標本番号、縦軸を合計残差として、データを可視化したものが以下になります。
これを見ると意図的に経年劣化させたFPGAの残差合計が1番小さくなっている．
なんで？むしろ高くなって欲しかったのになぜか小さくなった．

次にヒストグラムにしてみる
"""

# plt.figure()
# plt.hist(data, bins=25)
# plt.title("histgram")
# plt.xlabel("residual")
# plt.ylabel("frequency")
# plt.show()

"""
ここから異常値を見つけたい
ぱっと見わからない
統計学的にせめていく
観測データがN個あるとき、データをまとめて Dという記号で表すとします。
D={x1,x2,...,xN}

ここで、観測データが多次元の場合は、xが列ベクトルとして表されます。
また、このデータの中には異常な観測データが含まれていないか、含まれていたとしてもその影響は無視できると仮定します。
これから前述した統計的異常検知の3ステップに沿って話を進めていきます。


① 未知パラメータを含む確率分布を仮定
　上記のヒストグラムによると、体重データの分布は若干左右非対称ではありますが、おおむね山の形になっています。
　そこで、それぞれの観測データxが平均μ、分散 σ2 の正規分布に従うと仮定します。

N(x|μ,σ2)=12πσ2‾‾‾‾‾√exp{−(x−μ)22σ2}

② データから未知パラメータを推定
　正規分布に含まれているパラメータは平均 μ、分散 σ2 の二つです。
言い換えると、この二つのパラメータが決定すると、分布の形が一意に決まるということです。
これらのパラメータを観測データから推定します。
パラメータ推定には最尤推定を用います。
最尤推定は、「観測データが得られる確率」をパラメータの関数とみなした尤度関数を最大化するようにパラメータを決定する推定法です。

2.
一般に、異常度の定義として、負の対数尤度を採用されることが多いです。
（これは情報理論におけるシャノン情報量と呼ばれるものです）

異常度は、
a(x′)=(x′−μ̂ σ̂ )2
という形で表されます。
F統計量とも呼ばれてるっぽい

F統計量を計算する関数を作りヒストグラムで表してみる
3.
異常度が決まると、それに閾値を設定することで異常判定をすることができます。
閾値は経験に基づき主観的に決める方法もありますが、できるだけ客観的基準に基づいて決めるのが望ましいです。

ここでホテリング理論が登場
観測データが正規分布に従うと仮定した元で異常検知を行う古典的手法
広く応用されていると
ホテリング理論が有効な理由に，異常検知が従う確率分布を明示的に導くことができる点．

これで異常度の確率分布がわかれば，それに基づいて闘値を決定することは簡単

ホテリング理論によると異常度 a(x′) はデータ数Nが十分に大きい時、自由度1のカイ二乗分布に従うということが数学的に証明できます。
"""


# # 標本平均
# mean = mean(data)

# # 標本分散
# variance = variance(data)

# # 異常度
# anomaly_scores = []
# for x in data:
#     anomaly_score = (x - mean)**2 / variance
#     anomaly_scores.append(anomaly_score)

# # カイ二乗分布による1%水準の閾値
# threshold = stats.chi2.interval(0.95, 1)[1]

# # 結果の描画
# num = np.arange(1, 53)
# plt.plot(num, anomaly_scores, "o", color = "b")
# plt.plot([0,53],[threshold, threshold], 'k-', color = "r", ls = "dashed")
# plt.xlabel("Sample number")
# plt.ylabel("Anomaly score")
# plt.show()


"""
次はそれぞれのFPGAの中での分布を調べてみる
そこでの異常値を調べて
調べてどうするんだよ

"""






# y = 11
# x = 5
# fig,ax = plt.subplots(y,x,figsize=(20,20))
# count = 0
# for i in range(y):
#     for j in range(x):
#         if count >= 52:
#             break
#         dummy = sns.distplot(data[count], ax=ax[i, j])
#         count += 1
#     if count >= 52:
#         break

# plt.show()

# def generate_residual_data(data_n, data):
#     """
#     測定値-推定値(周りの平均)
#     残差を求めてデータ生成
#     """
#     tmp_x = [0, 1, 0, -1]
#     tmp_y = [-1, 0, 1, 0]

#     residual_data = np.zeros_like(data)

#     for i in range(data_n):
#         for j in range(data[i].shape[0]):
#             for k in range(data[i].shape[1]):
#                 data_list = []
#                 for l in range(4):
#                     next_y = j + tmp_y[l]
#                     next_x = k + tmp_x[l]
#                     if 0 <= next_y < 148 and 0 <= next_x < 33:
#                         data_list.append(data[i, next_y, next_x])

#                 data_mean = mean(data_list)
#                 residual_data[i, j, k] = np.abs(data[i, j, k] - data_mean)

#     return residual_data

# residual_data, aged_residual_data = generate_nnr()

# data = []
# for i in range(50):
#     data.append(residual_data[i].flatten())
    
# for i in range(2):
#     data.append(aged_residual_data[i].flatten())
    
# data = np.array(data)

# y = 11
# x = 5
# fig,ax = plt.subplots(y,x,figsize=(20,20))
# count = 0
# for i in range(y):
#     for j in range(x):
#         if count >= 52:
#             break
#         dummy = sns.distplot(data[count], ax=ax[i, j])
#         count += 1
#     if count >= 52:
#         break

# plt.show()
#file = open('new.csv', 'w')    #既存でないファイル名を作成してください
#w = csv.writer(file)
#w = w.writerows(data)
# 
#file.close()
#
#dataset = pd.read_csv("./new.csv", header=None)
#dataset


# residual_data, aged_residual_data = generate_nnr()

# data = []
# for i in range(6):
#     data.append(residual_data[i].flatten())
    
# for i in range(2):
#     data.append(aged_residual_data[i].flatten())
    
# data = np.array(data)

# y = 3
# x = 3
# fig,ax = plt.subplots(y,x,figsize=(20,20))
# count = 0
# for i in range(y):
#     for j in range(x):
#         if count >= 8:
#             break
#         dummy = sns.distplot(data[count], ax=ax[i, j])
#         count += 1
#     if count >= 8:
#         break

# plt.show()

                      
#file = open('new.csv', 'w')    #既存でないファイル名を作成してください
#w = csv.writer(file)
#w = w.writerows(data)
# 
#file.close()
#
#dataset = pd.read_csv("./new.csv", header=None)
#dataset


# y = 3
# x = 3
# fig,ax = plt.subplots(y,x,figsize=(20,20))
# count = 0
# for i in range(y):
#     for j in range(x):
#         if count >= 8:
#             break
#         ax[i, j].hist(data[count])
#         ax[i, j].set_title(f'data{i},{j}')
#         count += 1
#     if count >= 8:
#         break

# plt.show()


# residual_data, aged_residual_data = generate_nnr()

#data = []
#for i in range(50):
#    data.append(residual_data[i].flatten())
#    
#for i in range(2):
#    data.append(aged_residual_data[i].flatten())
#    
#data = np.array(data)
#
#print(data.shape)
    

# data = []
# for i in range(6):
#     data.append(residual_data[i].flatten())
    
# for i in range(2):
#     data.append(aged_residual_data[i].flatten())
    
# data = np.array(data)

# new_data = []    
# for i in range(data.shape[0]):
#     tmp = []
#     for j in range(len(data[i])):
#         if data[i, j] >= 15:
#             tmp.append(data[i, j])
#     new_data.append(tmp)


# y = 3
# x = 3
# fig,ax = plt.subplots(y,x,figsize=(20,20))
# count = 0
# for i in range(y):
#     for j in range(x):
#         if count >= 8:
#             break
#         dummy = sns.distplot(new_data[count], ax=ax[i, j])
#         count += 1
#     if count >= 8:
#         break

# plt.show()