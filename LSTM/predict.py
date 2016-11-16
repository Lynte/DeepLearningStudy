%matplotlib inline
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import report, training, Chain, datasets, iterators, optimizers
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import matplotlib.pyplot as plt

results = [0]
x = np.linspace(0, 2 * np.pi * 3, 100)
for i in range(10):
    y = model.predictor(results[i])
    results.append(y.data)

plt.plot(t, results)
plt.show()
