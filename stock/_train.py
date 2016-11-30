import argparse
import numpy as np
import pandas as pd

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import matplotlib.pyplot as plt

from model import LSTMmodel


class LossFuncL(Chain):

    def __init__(self, predictor):
        super(LossFuncL, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        x.data = x.data.reshape((-1, 1)).astype(np.float32)
        t.data = t.data.reshape((-1, 1)).astype(np.float32)

        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        report({'loss': loss}, self)
        return loss


class LSTM_Iterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size=10, bprop_len=5, repeat=True):
        self.bprop_len = bprop_len
        self.dataset = dataset
        self.nsamples = len(dataset)
        self.batch_size = batch_size
        self.repeat = repeat

        self.epoch = 0
        self.iteration = 0

        self.offsets = [i * self.nsamples //
                        batch_size for i in range(batch_size)]

        self.is_new_epoch = False

    def __next__(self):
        if not self.repeat and self.iteration * self.batch_size >= self.nsamples:
            raise StopIteration
        x, t = self.get_data()
        self.iteration += 1

        epoch = self.iteration * self.batch_size // self.nsamples
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(x, t))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / self.nsamples

    def get_data(self):
        tmp0 = [self.dataset[(offset + self.iteration) % self.nsamples][1]
                for offset in self.offsets]
        tmp1 = [self.dataset[(offset + self.iteration + 1) % self.nsamples][1]
                for offset in self.offsets]
        return tmp0, tmp1

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class LSTM_updater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(LSTM_updater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = train_iter.bprop_len

    def uptate_core(self):
        loss = 0

        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        for i in range(self.seq_length):
            batch = np.array(train_iter.__next__()).astype(np.float32)
            x, t = batch[:, 0].reshape((-1, 1)), batch[:, 1]
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.zerograds()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()


def make_data(filename, dataname,  train_ratio):
    df = pd.read_csv(filename)
    data = (df[dataname]).as_matrix()
    train_size = int(len(data) * train_ratio)
    return list(enumerate(data[:train_size])), list(enumerate(data[train_size:]))


def get_dataset(N, N_Loop):
    x = np.linspace(0, 2 * np.pi * N_Loop, N)
    y = np.sin(x)

    return list(zip(x, y))


def main():
    parser = argparse.ArgumentParser(description='learn sin wave')
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()

    model = LossFuncL(LSTMmodel(1, 5, 1))
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    #train = get_dataset(100, 3)
    #test = get_dataset(100, 3)
    train, test = make_data('data/nikkei_225.csv', 'close', 0.8)

    train_iter = LSTM_Iterator(train, batch_size=10, bprop_len=10)
    test_iter = LSTM_Iterator(test, batch_size=10, bprop_len=10, repeat=False)

    updater = LSTM_updater(train_iter, optimizer, args.gpu)
    trainer = training.Trainer(updater, (100, 'epoch'), out='result')

    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    trainer.extend(extensions.Evaluator(
        test_iter, eval_model, device=args.gpu,
        eval_hook=lambda _: eval_rnn.reset_state()
    ))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss']
        )
    )
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.run()
    chainer.serializers.save_npz('lstmmodel.model', model)
    chainer.serializers.save_npz('lstmstate.state', optimizer)

if __name__ == '__main__':
    main()
