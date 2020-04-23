import numpy as np
import matplotlib.pyplot as plt

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
    k_means(data, [2, 20])

    print()
    #(2) [2, 3, 10]
    k_means(data, [2, 3, 10])
    
    print()
    #(3) [12, 25, 30]
    k_means(data, [12, 25, 30])

if __name__ == "__main__":
    main()
    pass