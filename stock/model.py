import chainer
import chainer.functions as F
import chainer.links as L

class LSTMmodel(chainer.Chain):
    def __init__(self, n_input, n_unit, n_output):
        super(LSTMmodel, self).__init__(
                l1 = L.Linear(n_input, n_unit),
                lstm = L.LSTM(n_unit, n_unit),
                l2 = L.Linear(n_unit, n_output),
                )

    def reset_state(self):
        self.lstm.reset_state()

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = self.lstm(h1)
        return self.l2(h2)


