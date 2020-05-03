import numpy as np

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
    distance[0] = np.sum(data - centers[0])

    for k in range(1, cluster_num):
        np.random.seed(seeds*(k+1))
        probability = data / np.sum(distance)
        probability /= probability.sum() #正規化　probabilities do not sum to 1　と怒られたから

        centers[k] = np.random.choice(data, 1, p=probability)
        distance[k] = np.sum(data - centers[k])

    return centers


import matplotlib.pyplot as plt


def wrap_k_means(data, k=3):
    cluster = kmeans_plus_plus(data, k) # 初期値kmeans++
    k_means(data, cluster)

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
            min_d = 100000 # てきとうに大きい数
            for j,c in enumerate(cluster):
                dist = abs(d - c)
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
        print("{}回目".format(count))
        print("data   : ", data)
        print("prof   : ", prof)
        #print("old    : ", old_cluster) 
        print("cluster: ", cluster)
        
        # 収束チェック
        for i,c in enumerate(cluster):
            if c != old_cluster[i]:
                conv = True
                break
            else:
                conv = False

    # 結果出力
    print("result")
    print("data   : ", data) # debug
    print("prof   : ", prof) # debug
    print("cluster: ", cluster) # debug    
    
def main():
    data = np.array([2, 3, 4, 10, 11, 12, 20, 25, 30], dtype=float)
    #(1) [2, 20]
    wrap_k_means(data, 2)

if __name__ == "__main__":
    main()
    pass