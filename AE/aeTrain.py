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
import chainer
from chainer import cuda, Variable, FunctionSet, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from aeModel import Autoencoder

xp = cuda.cupy

def plot_mnist_data(samples, epoch, dir):
    for index, (data, label) in enumerate(samples):
        plt.subplot(4, 4, index + 1)
        plt.axis('off')
        plt.imshow(data.reshape(28, 28), cmap=cm.gray_r, interpolation='nearest')
        n = int(label)
        plt.title(n, color='red')
    plt.savefig(dir+"/epoch_"+str(epoch)+'.png')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=20000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--pict_dir', type=str, default='pict')
    parser.add_argument('--output_interval', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.pict_dir):
        os.mkdir(args.pict_dir)
    #set values
    n_batch = args.batch
    train_size = args.train_size
    test_size = args.test_size
    n_epoch = args.epoch
    #get datas
    train, test = chainer.datasets.get_mnist()
    train = train[0:train_size]
    train = [data[0] for data in train]

    fordraw = test[0:16]
    test = test[0:test_size]
    test = [data[0] for data in test]

    #make model
    model = Autoencoder()
    model.compute_accuracy = False
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    for epoch in range(n_epoch):
        print("epoch{}".format(epoch+1))
        perm = np.random.permutation(train_size)
        #train loop
        total_loss = 0
        for i in range(0, train_size-n_batch, n_batch):
            x_batch = xp.asarray([train[perm[i+j]] for j in range(n_batch)])
            y_batch = x_batch.copy()

            optimizer.zero_grads()
            loss = F.mean_squared_error(model(x_batch), y_batch)
            loss.backward()
            optimizer.update()

            total_loss += float(loss.data)*n_batch

        print("\ttrain mean loss:{}".format(total_loss/train_size))

        #evaluate loop
        total_loss = 0
        for i in range(0, test_size-n_batch, n_batch):
            x_batch = xp.asarray(test[i:i+n_batch])
            y_batch = x_batch.copy()

            loss = F.mean_squared_error(model(x_batch), y_batch)

            total_loss += float(loss.data)*n_batch

        print("\ttest mean loss:{}".format(total_loss/test_size))


        if not (epoch % args.output_interval) :
            pred_list = []
            for (data, label) in fordraw:
                pred = model(xp.asarray(data).reshape(1,784)).data
                pred_list.append((cuda.to_cpu(pred), label))
            plot_mnist_data(pred_list, epoch, args.pict_dir)

    serializers.save_npz(args.checkpoint_dir+'/ae.model', model)
    serializers.save_npz(args.checkpoint_dir+'/mlp.state', optimizer)

if __name__ == '__main__':
    main()
