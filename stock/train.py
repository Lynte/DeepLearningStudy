import numpy as np
import pandas as pd

#make train data and test data
def make_data(filename, dataname,  train_ratio):
    df = pd.read_csv(filename)
    data = (df[dataname]).as_matrix()
    train_size = int(len(data)*train_ratio)
    return data[:train_size], data[train_size:]

train, test = make_data('data/nikkei_225.csv', 'close', 0.8)
print('train size:\t{}\ntest size:\t{}'.format(len(train), len(test)))

