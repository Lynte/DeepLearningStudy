import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

#define model
class Autoencoder(chainer.Chain):
    def __init__(self):
        super.__init__(Autoencoder, self):
            encoder = L.Linear(784, 64),
            decoder = L.Linear(64, 784),
            )
    def __call__(self, x, hidden=False):
        h = F.relu(self.encoder(x))
        if hidden:
            return h
        else:
            return F.relu(self.decoder(h))

