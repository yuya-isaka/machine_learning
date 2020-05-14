import numpy as np
import pandas as pd

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

    tmp_x = [-1, 0, 1, 1, 1, 0, -1, -1]
    tmp_y = [-1, -1, -1, 0, 1, 1, 1, 0]
    residual_data = np.zeros_like(data)
    for i in range(52):
        for j in range(148):
            for k in range(33):
                if data[i, j, k] != 0:
                    data_list = []
                    for l in range(8):
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

    count_list = []
    for d in data:
        counter = 0
        for i in d:
            if i <= 1:
                counter += 1
        count_list.append(counter)

    print(count_list)



if __name__ == "__main__":
    main()
    pass