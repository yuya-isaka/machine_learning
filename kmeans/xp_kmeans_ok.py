import sys
import os
sys.path.append(os.pardir)
from dataset.fpga_fp_data import generate_data
import matplotlib.pyplot as plt              
import numpy as np
import pandas as pd
import copy
from statistics import mean
import pickle

np.seterr(divide='ignore', invalid='ignore')


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


def generate_data_kmeans(directory, n, aged_n):
    """
    csvデータを綺麗に整えて返す．
    (4884, 3)
    データ数4884, 特徴量3
    """
    
    new_data, aged_data = generate_data(directory, n, aged_n)
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
    
    seeds = 50
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
    sseは今回は使っていない
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
            min_value = tmp_value
            neighbor_point = i

    return neighbor_point


def silhouette(centers, cluster_point, neighbor_point, renovate_data, idx_list):
    """
    クラスタの平均sviを返す
    """

    cluster_renovate_data = np.delete(renovate_data[idx_list == cluster_point, :], 3, 1)
    neighbor_renovate_data = np.delete(renovate_data[idx_list == neighbor_point, :], 3, 1)

    svi_list = []

    for i in range(cluster_renovate_data.shape[0]):
        #i番目の周波数点のsvi
        oci_list = []
        nci_list = []
        for j in range(cluster_renovate_data.shape[0]):
            oci_list.append(np.sum((cluster_renovate_data[i] - cluster_renovate_data[j])**2))
        for j in range(neighbor_renovate_data.shape[0]):
            nci_list.append(np.sum((cluster_renovate_data[i] - neighbor_renovate_data[j])**2))

        oci = mean(oci_list)
        nci = mean(nci_list)

        svi = (nci - oci)/max(nci, oci)

        svi_list.append(svi)

    svi_mean = mean(svi_list)

    return svi_mean


def main(search_FPGA_num, aged_search_FPGA_num, cluster_num):

    data, aged_data, csv_data, csv_aged_data = generate_data_kmeans('fresh_aged_ieice', search_FPGA_num, aged_search_FPGA_num)

    acn_list = []
    aged_acn_list = []

    for fpga_num in range(search_FPGA_num):

        acn = 0
        acn_value = -float('inf')

        for k in range(2, cluster_num+1):
            idx_list, _, centers = kmeans(data[fpga_num], k)
            renovate_data = renovate(idx_list, csv_data[fpga_num])

            svi_mean_list = []
            for i in range(k):
                neighbor_point = center_neighbor(centers, i)
                tmp_svi_mean = silhouette(centers, i, neighbor_point, renovate_data, idx_list)
                svi_mean_list.append(tmp_svi_mean)
                print('FPGA:{}, cluster:{}, silhouette:{}'.format(fpga_num+1, k, i+1))

            svi_mean = mean(svi_mean_list)

            if svi_mean > acn_value:
                acn_value = svi_mean
                acn = k

        acn_list.append(acn)
        f = open('acn_list.binaryfile', 'wb')
        pickle.dump(acn_list, f)
        f.close()

    for aged_fpga_num in range(aged_search_FPGA_num):

        aged_acn = 0
        aged_acn_value = -float('inf')

        for k in range(2, cluster_num+1):
            aged_idx_list, _, aged_centers = kmeans(aged_data[aged_fpga_num], k)
            aged_renovate_data = renovate(aged_idx_list, csv_aged_data[aged_fpga_num])

            aged_svi_mean_list = []
            for i in range(k):
                aged_neighbor_point = center_neighbor(aged_centers, i)
                aged_tmp_svi_mean= silhouette(aged_centers, i, aged_neighbor_point, aged_renovate_data, aged_idx_list)
                aged_svi_mean_list.append(aged_tmp_svi_mean)
                print('aged_FPGA:{}, aged_cluster:{}, aged_silhouette:{}'.format(aged_fpga_num+1, k, i+1))

            aged_svi_mean = mean(aged_svi_mean_list)

            if aged_svi_mean > aged_acn_value:
                aged_acn_value = aged_svi_mean
                aged_acn = k

        aged_acn_list.append(aged_acn)
        f = open('aged_acn_list.binaryfile', 'wb')
        pickle.dump(aged_acn_list, f)
        f.close()

    for i, value in enumerate(acn_list):
        print('new_FPGA:{}, ACN{}'.format(i+1, value))
    for i, aged_value in enumerate(aged_acn_list):
        print('aged_FPGA:{}, aged_ACN{}'.format(i+1, aged_value))



main(search_FPGA_num=2, aged_search_FPGA_num=2, cluster_num=2)