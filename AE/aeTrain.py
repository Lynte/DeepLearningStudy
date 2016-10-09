import time
import math
import sys
import argparse
import copy
import os
import codecs

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import chainer.cuda.cupy as xp
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
import chainer.links as L
from aeModel import Autoencoder

def plot_mnist_data(samples, epoch):
    for index, (data, label) in enumerate(samples):
        plt.subplot(4, 4, index + 1)
        plt.axis('off')
        plt.imshow(data.reshape(28, 28), cmap=cm.gray_r, interpolation='nearest')
        n = int(label)
        plt.title(n, color='red')
    plt.savefig("./pict/epoch_"+str(epoch)+'.png')

def main():
    parser = parser.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=20000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    #set values
    n_batch = args.batch
    train_size = args.train_size
    test_size = args.test_size
    n_epoch = args.epoch
    #get datas
    train, test = chainer.datasets.get_mnist()
    train = train[0:train_size]
    train = [data[0] for data in train]

    fordraw = test[0:25]
    test = test[0:test_size]
    test = [data[0] for data in test]

    #make model
    model = Autoencoder(), loss_fun=F.mean_squared_error
    model.compute_accuracy = False
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    for epoch in range(n_epoch):
        print("epoch{}".format(epoch+1))
        perm = xp.random.permutation(train_size)

        #train loop
        total_loss = 0
        for i in range(0, train_size, n_batch):
            x_batch = train[i:i+n_batch]
            y_batch = x_batch.copy()

            optimizer.zero_grads()
            loss = F.mean_squared_error(model(x_batch), y_batch)
            loss.backword()
            optimizer.update()

            total_loss += loss*n_batch

        print("\ttrain mean loss:{}".format(total_loss/train_size))

        #evaluate loop
        total_loss = 0
        for i in range(0, test_size, n_batch):
            x_batch = test[i:i+n_batch]
            y_batch = x_batch.copy()

            loss = F.mean_squared_error(model(x_batch), y_batch)

            total_loss += loss*n_batch

        print("\ttest mean loss:{}".format(total_loss/test_size))

        if not (epoch % 20) :
            pred_list = []
            for (data, label) in fordraw:
                pred = model(data)
                pred_list.append((pred, label))
            plot_mnist_data(pred_list, epoch)

if __name__ == '__main__':
    main()
