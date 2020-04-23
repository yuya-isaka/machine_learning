import pickle
import os
import numpy as np
import pandas as pd

def csv_to_data(directory, data_n):
    data = []

    for i in range(1, data_n+1):
        tmp_data = pd.read_csv(directory+'/s'+str(i)+'.csv', header=None).values
        data.append(tmp_data)

    f = open('FPGA_data.binaryfile', 'wb')
    pickle.dump(data, f)
    f.close()

    return np.array(data)


def csv_to_aged_data(directory, aged_data_n):
    aged_data = []

    for i in range(1, aged_data_n+1):
        tmp_data = pd.read_csv(directory+'/s'+str(i)+'_aged.csv', header=None).values
        aged_data.append(tmp_data)

    f = open('FPGA_aged_data.binaryfile', 'wb')
    pickle.dump(aged_data, f)
    f.close()

    return np.array(aged_data)


def generate_data(directory, data_n, aged_data_n):
    """
    data(指定数，148，33)
    aged_data(指定数，148，33)
    """
    
    if os.path.isfile('FPGA_data.binaryfile'):
        f = open('FPGA_data.binaryfile', 'rb')
        data = pickle.load(f)
        data = np.array(data)
        f.close()
    else:
        data = csv_to_data(directory, data_n)

    if os.path.isfile('FPGA_aged_data.binaryfile'):
        f = open('FPGA_aged_data.binaryfile', 'rb')
        aged_data = pickle.load(f)
        aged_data = np.array(aged_data)
        f.close()
    else:
        aged_data = csv_to_aged_data(directory, aged_data_n)

    return data, aged_data
