import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import treelib

class IsolateTree:
    class __data:
        def __init__(self):
            self.feature_split = None
            self.threshhold_split = None
            self.n_samples = None

    def __init__(self):
        self.__tree = treelib.Tree()

    def __get_normalization_factor(self, n):
        return 2 * (np.log(n-1) + 0.57)

class IsolateForest:
    def __init__(self):
        self.__trees = []

    def fit(self,data,n_trees,n_subsamples):
        n_samples = data.shape[0]
        for _ in range(n_trees):
            sub_items = np.random.choice(n_samples, n_subsamples, replace=False)
            tree = IsolateTree()
            tree.fit(data[sub_items])
            self.__trees.append(tree)

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

    model = IsolateForest()
    model.fit(data, 1000, 15)

if __name__ == "__main__":
    main()
    pass