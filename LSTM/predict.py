import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import matplotlib.pyplot as plt
from model import LSTMmodel
from train import LossFuncL

model = LossFuncL(LSTMmodel(1, 5, 1))
optimizer = optimizers.Adam()
optimizer.setup(model)

chainer.serializers.load_npz('./lstmmodel.model', model)
chainer.serializers.load_npz('./lstmstate.state', optimizer)
last = np.array([0]).astype(np.float32)
results = np.array([])
x = np.linspace(0, 2 * np.pi * 6, 200)

model.predictor.reset_state()
for i in range(200):
    y = model.predictor(chainer.Variable(last.reshape((-1, 1))))
    if i < 15 :
        last = np.array([np.sin(x[i])]).astype(np.float32)
        results = np.append(results, last[0])
    else:
        last = y.data;
        results = np.append(results, last[0][0])

plt.ylim(-1, 1)
plt.plot(x, results)
plt.plot(x, np.sin(x))
plt.show()
