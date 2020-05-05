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
from sklearn.cluster import KMeans

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

    """９２０この値を消す"""
def delete_920(data, check):
    new = np.zeros_like(data)
    counter = 0
    for i in data:
        for j in range(i.shape[0]):
            for k in range(i.shape[1]):
                if [j, k] in check:
                    new[counter, j, k] = 0 
                else:
                    new[counter, j, k] = data[counter, j, k]
                    
        counter += 1
                    
    return new


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


"""リストを繋げる関数"""
def connect(a, b):
    tmp = []
    for i in range(a.shape[0]):
        tmp.append(a[i])
        
    for i in range(b.shape[0]):
        tmp.append(b[i])
    
    tmp = np.array(tmp)
    return tmp

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

    newdata = connect(fresh, aged) # (52, 148, 33)
    check_FPGA = []
    for i in range(148):
        for j in range(33):
            if fresh[0, i, j] == 0:
                check_FPGA.append([i,j])
    freshaged = delete_920(newdata, check_FPGA) #(52, 148, 33)

    new_freshaged = change_flatten(freshaged) # (52 4884)

    nonzero_freshaged = delete_zero(new_freshaged) #(52, 3964)

    new = []
    for i in nonzero_freshaged:
        tmp = []
        for j in range(len(i)):
            tmp.append([i[j], 0])

        new.append(tmp)

    new = np.array(new)

    every_cluster_list = [] # (7, 52)
    for x in range(2, 9):

        fpga_class = []
        for i in range(new.shape[0]):
            kmeans_model = KMeans(n_clusters=x, init='k-means++').fit(new[i])
            prof = kmeans_model.labels_
            cluster = kmeans_model.cluster_centers_[0]

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

    f = open('ACN_list_5.binaryfile', 'wb')
    pickle.dump(every_cluster_list, f)
    f.close()


if __name__ == "__main__":
    main()
    pass


