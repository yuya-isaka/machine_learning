import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
np.random.seed(100)

def main():
    data = []
    for i in range(1, 51):
        tmp_data = pd.read_csv('fresh_aged_ieice/s'+str(i)+'.csv', header=None).values
        data.append(tmp_data)
    for i in range(1, 3):
        tmp_data = pd.read_csv('fresh_aged_ieice/s'+str(i)+'_aged.csv', header=None).values
        data.append(tmp_data)
    data = np.array(data)

    check = []
    for i in range(148):
        for j in range(33):
            if data[0, i, j] == 0:
                check.append([i,j])
    for i in range(52):
        for j in range(148):
            for k in range(33):
                if [j,k] in check:
                    data[i, j, k] = 0

    tmp_1 = []
    for i in range(52):
        tmp_2 = data[i].flatten()
        tmp_1.append(tmp_2[tmp_2 != 0])
    data = np.array(tmp_1)

    tmp_1 = []
    for d in data:
        tmp_2 = []
        for i in range(len(d)):
            tmp_2.append([d[i], 0])
        tmp_1.append(tmp_2)
    data = np.array(tmp_1)

    SVI_list = []
    for i in range(data.shape[0]):
        tmp = []
        for k in range(2, 9):
            km = KMeans(n_clusters=k,    # クラスターの個数
                init='k-means++',        # セントロイドの初期値をランダムに設定
                n_init=10,               # 異なるセントロイドの初期値を用いたk-meansアルゴリズムの実行回数
                max_iter=300,            # k-meansアルゴリズムの内部の最大イテレーション回数
                tol=1e-04,               # 収束と判定するための相対的な許容誤差
                random_state=0)          # セントロイドの初期化に用いる乱数発生器の状態
            model = km.fit_predict(data[i])
            score = silhouette_score(data[i], model)
            tmp.append(score)
            print(f'{k}回目')
        print(f'[{i+1}回目]')
        SVI_list.append(tmp)

    index_list = np.arange(1,53)
    acn_list = []
    average_list = []
    status_list = []
    for svi in SVI_list:
        acn = np.argmax(svi) + 2
        average = svi[acn-2]
        acn_list.append(acn)
        average_list.append(average)
    for i in range(50):
        status_list.append('unused')
    for i in range(2):
        status_list.append('aged')

    table = [average_list, acn_list, status_list]
    table = np.array(table).T
    df = pd.DataFrame(table,
                    columns=['Maximum Average Silhouette value', 'Appropriate Cluster Number', 'Status'],
                    index=index_list)
    print(df)


if __name__ == "__main__":
    main()
    pass