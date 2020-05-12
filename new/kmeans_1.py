import numpy as np
import pandas as pd
np.random.seed(69)

def kmeans(data, k):
    """kmeans++"""
    feature_num = data.shape[0]
    centers = np.zeros(k)
    distance = np.zeros(feature_num)
    probability = np.repeat(1/feature_num, feature_num)
    centers[0] = np.random.choice(data, 1, p=probability)
    for i in range(feature_num):
        distance[i] = np.abs(data[i] - centers[0]) ** 2

    for i in range(1, k):
        probability = []
        for j in range(feature_num):
            probability.append(distance[j]/np.sum(distance))

        centers[i] = np.random.choice(data, 1, p=probability)
        for j in range(feature_num):
            tmp_1 = float('inf')
            for k in range(i):
                tmp_2 = np.abs(data[j] - centers[k]) ** 2 
                if tmp_1 >= tmp_2:
                    tmp_1 = tmp_2
            distance[j] = tmp_1

    """kmeans"""
    data = np.array(data, dtype=float)
    index_list = np.zeros(len(data))
    centers = np.sort(np.array(centers, dtype=float))
    old_centers = np.zeros(len(centers))

    flag = True 
    while flag:
        # index割り当て
        for i, d in enumerate(data):
            min_d = float('inf')
            for j, c in enumerate(centers):
                dist = (abs(d - c))**2
                if min_d > dist:
                    min_d = dist
                    index_list[i] = j

        # center更新
        for i, c in enumerate(centers):
            m = 0; n = 0
            for j, il in enumerate(index_list):
                if il == i:
                    m += data[j]
                    n += 1
            if m != 0:
                m /= n
                old_centers[i] = c
                centers[i] = m

        # 収束チェック
        for i, c in enumerate(centers):
            if c != old_centers[i]:
                flag = True
                break
            else:
                flag = False

    return index_list, centers

def silhouette(data, index_list, center_num):
    SVI_list = []
    # データiのOCIとNCIから,SVIを求める
    for i in range(len(data)): # データ
        OCI = 0
        NCI = []
        # データiとの他のデータ点との距離を求める
        for j in range(center_num): # センターごとに求めていく
            tmp_list_1 = []
            for k in range(len(data)):  
                if index_list[k] == j:
                    tmp_list_1.append(data[k])
            tmp_list_2 = []
            for k in tmp_list_1:
                tmp_list_2.append(abs(data[i] - k)**2)

            if index_list[i] == j:
                OCI = np.sum(np.array(tmp_list_2)) / (len(tmp_list_2) - 1)
            else:
                NCI.append(np.mean(np.array(tmp_list_2)))

        NCI.sort()
        svi = (NCI[0] - OCI) / max(NCI[0], OCI)
        SVI_list.append(svi)

    return np.mean(np.array(SVI_list))


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

    SVI_list = []
    for i in range(52):
        svi_list = []
        for j in range(2,9):
            index_num, centers = kmeans(data[i], j)
            tmp = silhouette(data[i], index_num, len(centers))
            svi_list.append(tmp)
            print(j)
        SVI_list.append(svi_list)
        print(i+1)

    ACN_list = []
    ACN_index_list = []
    STATUS_list = []
    for i in range(52):
        tmp = np.argmax(SVI_list[i])
        ACN_list.append(SVI_list[i][tmp])
        ACN_index_list.append(tmp+2)
    for i in range(50):
        STATUS_list.append('unused')
    for i in range(2):
        STATUS_list.append('aged')

    table = [ACN_list, ACN_index_list, STATUS_list]
    table = np.array(table).T
    df = pd.DataFrame(table, columns=['Maximum Average Silhouette value', 'Appropriate Cluster Number', 'Status'], index=np.arange(1,53))
    print(df)



if __name__ == "__main__":
    main()
    pass