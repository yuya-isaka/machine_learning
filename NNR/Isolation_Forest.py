import numpy as np 
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt 

def main():
    data_n = 1000
    args = {
        'n_samples': data_n,
        'n_features': 2,
        'centers': 1,
        'random_state': 42,
    }

    x_normal, y_normal = make_blobs(**args)

    outlier_n = 200
    distribution_range = 6
    x_outlier = np.random.uniform(low=-distribution_range, high=distribution_range,size=(outlier_n, 2))

    y_outlier = np.ones((outlier_n,))

    x_outlier += x_normal.mean(axis=0)

    x = np.concatenate([x_normal, x_outlier], axis=0)
    y = np.concatenate([y_normal, y_outlier], axis=0)

    plt.scatter(x[y == 0, 0],
                x[y == 0, 1], 
                label='negative')
    plt.scatter(x[y == 1, 0],
                x[y == 1, 1], 
                label='positive')

    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()