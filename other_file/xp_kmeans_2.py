import matplotlib.pyplot as plt              
import numpy as np
import pandas as pd
import copy
from statistics import mean

np.seterr(divide='ignore', invalid='ignore')

def csv_to_data(directory, n, aged_n):
    """ 欲しいcsvファイルの数を指定したら，綺麗にしたデータを返してくれる関数
    argument
    directory: csvファイルがあるディレクトリ名
    n: 欲しいFPGAのデータの数(0 - 50)
    aged_n: 欲しいAged_FPGAのデータの数(0 - 2)

    return
    data: 取り出したFPGAのcsvデータ(148, 33)
    aged_n: 取り出したAged_FPGAのデータ(148, 33)
    """
    new_data = []
    aged_data = []

    for i in range(1, n+1):
        tmp = pd.read_csv(directory+'/s'+str(i)+'.csv', header=None).values
        new_data.append(tmp)

        if aged_n and 1 <= i <= aged_n:
            tmp = pd.read_csv(directory+'/s'+str(i)+'_aged.csv', header=None).values
            aged_data.append(tmp)

    return np.array(new_data), np.array(aged_data)


def change_data(new_data, aged_data):
    """データを与えたらkmeansに突っ込むようのデータを返してくれる関数
    argument
    new_data: 新しいFPGAの整ったデータ
    aged_data: 経年劣化FPGAの整ったデータ

    return
    renew_data: kmeansに突っ込む用の新しいFPGAデータ
    reaged_data: kmeansに突っ込む用の経年劣化FPGAデータ
    """
    renew_data = []
    reaged_data = []

    for i in range(len(new_data)):
        tmp = new_data[i]
        tmp_list = []
        for j in range(tmp.shape[0]):
            for k in range(tmp.shape[1]):
                tmp_list.append([j, k, tmp[j, k]])
        renew_data.append(tmp_list)

    if aged_data is not False:
        for i in range(len(aged_data)):
            tmp = aged_data[i]
            tmp_list = []
            for j in range(tmp.shape[0]):
                for k in range(tmp.shape[1]):
                    tmp_list.append([j, k, tmp[j, k]])
            reaged_data.append(tmp_list)

    return np.array(renew_data), np.array(reaged_data)


def generate_data(directory, n, aged_n):
    """
    csvデータを綺麗に整えて返す．
    (4884, 3)
    データ数4884, 特徴量3
    """
    new_data, aged_data = csv_to_data(directory, n, aged_n)
    renew_data, reaged_data = change_data(new_data, aged_data)

    return renew_data, reaged_data, new_data, aged_data




def kmeans_algorithm(data, cluster_num, centers):
    """
    データと，クラスター数と，中心点の位置を与えると
    クラスタに分けたインデックスリスト，それぞれのクラスタのsse，中心点の位置を返す
    """
    idx_list = np.zeros(data.shape[0])
    sse_list = np.zeros(cluster_num)

    tmp_centers = np.zeros_like(centers)
    while 1:
        if (tmp_centers == centers).all():
            break
        tmp_centers = copy.deepcopy(centers)

        for i in range(data.shape[0]):
            idx_list[i] = np.argmin(np.sum((data[i, :] - centers)**2, axis=1))

        for k in range(cluster_num):
            centers[k, :] = data[idx_list == k, :].mean(axis=0)
            sse_list[k] = np.sum((data[idx_list == k, :]-centers[k, :])**2)

    return idx_list, sse_list, centers


def kmeans_plus_plus(data, cluster_num):
    """
    中心点ゲット！
    """
    seeds = 39
    np.random.seed(seeds)

    data_num = data.shape[0]
    feature_num = data.shape[1]

    distance = np.zeros(data_num*cluster_num).reshape(data_num, cluster_num)
    centers = np.zeros(feature_num*cluster_num).reshape(cluster_num, feature_num)

    probability = np.repeat(1/data_num, data_num)
    centers[0, :] = data[np.random.choice(np.arange(data_num), 1, p=probability)]
    distance[:, 0] = np.sum((data - centers[0, :])**2, axis=1)

    for k in range(1, cluster_num):
        np.random.seed(seeds*(k+1))
        probability = np.sum(distance, axis=1) / np.sum(distance)

        centers[k, :] = data[np.random.choice(np.arange(data_num), 1, p=probability)]
        distance[:, k] = np.sum((data-centers[k, :])**2, axis=1)
    return centers


def kmeans(data, cluster_num):
    """
    データをkmeansアルゴリズムでクラスタリング
    """

    centers = kmeans_plus_plus(data, cluster_num)

    idx_list, sse_list, centers = kmeans_algorithm(data, cluster_num, centers)

    return idx_list, sse_list, centers


def renovate(idx_list, csv_data):
    """
    データを復元
    """
    renovate_data = []

    y = 0
    x = 0

    for i in range(csv_data.shape[0]*csv_data.shape[1]):
        if y == (csv_data.shape[0]):
            break

        renovate_data.append([y, x, csv_data[y, x], idx_list[i]])
        x += 1
        if x == (csv_data.shape[1]):
            x = 0
            y += 1

    return np.array(renovate_data)


def center_neighbor(centers, cluster_point):
    """
    中心点と近隣のクラスタを教えてくれる
    """
    neighbor_point = 0
    min_value = float('inf')
    for i in range(centers.shape[0]):
        if(i == cluster_point):
            continue
        tmp_value = np.sum((centers[cluster_point, :] - centers[i, :])**2)
        if min_value > tmp_value:
            neighbor_point = i
            min_value = tmp_value

    return neighbor_point


def silhouette(centers, cluster_point, neighbor_point, renovate_data, idx_list):
    """
    クラスタの平均sviを返す
    """
    # deleted_data = np.delete(renovate_data, 3, 1)

    # cluster_neighbor = 0
    # for i in range(renovate_data.shape[0]):
    #     if renovate_data[i][3] == neighbor_point:
    #         cluster_neighbor += np.sum((centers[neighbor_point, :] - deleted_data[i, :])**2)

    # svi = sse_list[cluster_point] - cluster_neighbor
    # return svi

    cluster_renovate_data = np.delete(renovate_data[idx_list == cluster_point, :], 3, 1)
    neighbor_renovate_data = np.delete(renovate_data[idx_list == neighbor_point, :], 3, 1)

    svi_list = []
    count = 0

    for i in range(cluster_renovate_data.shape[0]):
        print('svi_function: {}'.format(count))
        oci_list = []
        for j in range(cluster_renovate_data.shape[0]):
            oci_list.append(np.sum((cluster_renovate_data[i] - cluster_renovate_data[j])**2))
        nci_list = []
        for j in range(neighbor_renovate_data.shape[0]):
            nci_list.append(np.sum((cluster_renovate_data[i] - neighbor_renovate_data[j])**2))

        oci = mean(oci_list)
        nci = mean(nci_list)

        svi_list.append((nci - oci)/max(oci, nci))
        count += 1

    svi_mean = mean(svi_list)
    print(svi_list)

    return svi_mean

def main():
    search_FPGA_num = 1
    search_Aged_FPGA_num = 0
    cluster_num = 2

    data, aged_data, csv_data, csv_aged_data = generate_data('fresh_aged_ieice', search_FPGA_num, search_Aged_FPGA_num)

    idx_list, sse_list, centers = kmeans(data[search_FPGA_num-1], cluster_num)

    renovate_data = renovate(idx_list, csv_data[search_FPGA_num-1])
    
    neighbor_point = center_neighbor(centers, 0)

    svi_mean = silhouette(centers, 0, neighbor_point, renovate_data, idx_list)
    print(svi_mean)



main()
