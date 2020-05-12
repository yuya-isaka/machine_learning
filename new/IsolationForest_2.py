import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import treelib
np.random.seed(69)

class IsolateTree:
    class __data:
        def __init__(self):
            self.feature_split = None
            self.threshold_split = None
            self.n_samples = None

    def __init__(self):
        self.__tree = treelib.Tree()

    def fit(self, X):
        n_samples = X.shape[0]
        self.__c = self.__get_normalization_factor(n_samples)
        self.__max_height = np.round(np.log2(n_samples))
        root = self.__tree.create_node('root')
        self.__create_tree(root, X)

    def __get_normalization_factor(self, n):
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def __create_tree(self, parent, X):
        n_samples, n_features = X.shape
        data = self.__data()

        if n_samples == 0:
            data.n_samples = n_samples
            self.__tree.update_node(parent.identifier, data=data)
            return
        if self.__tree.depth() > self.__max_height or (X == X[0]).all():
            data.n_samples = n_samples
            self.__tree.update_node(parent.identifier, data=data)
            return

        data.feature_split = np.random.choice(n_features, 1)
        data.threshold_split = (max(X[:, data.feature_split]) - min(X[:, data.feature_split])) * np.random.random() + min(X[:, data.feature_split])
        self.__tree.update_node(parent.identifier, data=data)

        less_items = np.flatnonzero(X[:, data.feature_split] < data.threshold_split)
        greater_items = np.flatnonzero(X[:, data.feature_split] >= data.threshold_split)
        node = self.__tree.create_node('less ' + str(data.threshold_split), parent=parent)
        self.__create_tree(node, X[less_items])
        node = self.__tree.create_node('greater ' + str(data.threshold_split), parent=parent)
        self.__create_tree(node, X[greater_items])

    def get_abnormal_score(self, X):
        return 2 ** (-np.apply_along_axis(self.__get_path_length, 1, X, self.__tree.get_node(self.__tree.root)) / self.__c)

    def __get_path_length(self, x, node):
        if node.is_leaf():
            return self.__tree.depth(node.identifier) + (self.__get_normalization_factor(node.data.n_samples) if node.data.n_samples > 1 else 0)

        for child in self.__tree.children(node.identifier):
            if x[node.data.feature_split] < node.data.threshold_split and child.tag == 'less ' + str(node.data.threshold_split):
                return self.__get_path_length(x, child)
            elif x[node.data.feature_split] >= node.data.threshold_split and child.tag == 'greater ' + str(node.data.threshold_split):
                return self.__get_path_length(x, child)

class IsolateForest:
    def __init__(self):
        self.__trees = []

    def fit(self, X, n_trees, n_subsamples):
        n_samples = X.shape[0]
        for _ in range(n_trees):
            sub_items = np.random.choice(n_samples, n_subsamples, replace=False)
            tree = IsolateTree()
            tree.fit(X[sub_items])
            self.__trees.append(tree)

    def predict(self, X):
        n_samples = X.shape[0]
        abnormal_scores = np.zeros((n_samples, len(self.__trees)))
        for i in range(len(self.__trees)):
            abnormal_scores[:, i] = self.__trees[i].get_abnormal_score(X)
        return np.mean(abnormal_scores, axis=1)

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

    tmp_x = [0, 1, 0, -1]
    tmp_y = [-1, 0, 1, 0]
    residual_data = np.zeros_like(data)
    for i in range(52):
        for j in range(148):
            for k in range(33):
                if data[i, j, k] != 0:
                    data_list = []
                    for l in range(4):
                        next_y = j + tmp_y[l]
                        next_x = k + tmp_x[l]
                        if 0 <= next_y < 148 and 0 <= next_x < 33 and data[i, next_y, next_x] != 0:
                            data_list.append(data[i, next_y, next_x])
    
                    data_mean = np.mean(np.array(data_list))
                    residual_data[i, j, k] = abs(data[i, j, k] - data_mean)

    tmp_1 = []
    for i in range(52):
        tmp_2 = []
        for j in range(148):
            for k in range(33):
                if [j,k] in check:
                    continue
                else:
                    tmp_2.append(residual_data[i, j, k])
        tmp_1.append(tmp_2)
    data = np.array(tmp_1)

    print(data.shape)

    model = IsolateForest()
    sample_list = [10, 15, 20, 26, 30, 35, 40, 45, 52]
    rank = []
    for sample in sample_list:
        model.fit(data, 1000, sample) 
        result = model.predict(data)
        tmp = []
        for i in range(52):
            tmp.append([result[i], i])
        tmp.sort()
        tmp = np.array(tmp)[:,1]
        rank.append(tmp[47:52])
    rank = np.array(rank).T

    df = pd.DataFrame(rank, columns=sample_list, index=np.arange(1,6))
    print(df)

if __name__ == "__main__":
    main()
    pass