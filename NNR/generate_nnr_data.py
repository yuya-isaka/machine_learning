import sys
import os
sys.path.append(os.pardir)
from dataset.fpga_fp_data import generate_data
import matplotlib.pyplot as plt              
import numpy as np
import pandas as pd
from statistics import mean
import pickle

np.seterr(divide='ignore', invalid='ignore')


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
                residual_data[i, j, k] = data[i, j, k] - data_mean

    return residual_data


def load(name, data_n, data):
    """
    pickleファイルがあればそこから生成
    なければ作る
    """
    if os.path.isfile(name):
        f = open(name, 'rb')
        residual_data = pickle.load(f)
        residual_data = np.array(residual_data)
        f.close()
    else:
        residual_data = generate_residual_data(data_n, data)
        f = open(name, 'wb')
        pickle.dump(residual_data, f)
        f.close()

    return residual_data


def generate_nnr(data_n=50, aged_data_n=2):
    """
    残差集合のデータ生成
    """
    data, aged_data = generate_data('fresh_aged_ieice', data_n, aged_data_n)

    residual_data = load('residual_data.binaryfile', data_n, data)
    aged_residual_data = load('aged_residual_data.binaryfile', aged_data_n, aged_data)

    return residual_data, aged_residual_data


# data, aged_data = generate_nnr()

# print(data)
# print(aged_data)
# print(data.shape)
# print(aged_data.shape)