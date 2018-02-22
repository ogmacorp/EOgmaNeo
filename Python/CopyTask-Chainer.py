# ----------------------------------------------------------------------------
#  EOgmaNeo
#  Copyright(c) 2018 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the EOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import report
from chainer import Function
from chainer import optimizers


class RNN(chainer.Chain):
    def __init__(self, input_size, output_size, hidden_size=32):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.v = Variable(np.zeros((1, input_size), dtype=np.float32))
        self.h = Variable(np.zeros((1, hidden_size), dtype=np.float32))
        self.p = Variable(np.zeros((1, output_size), dtype=np.float32))

        super(RNN, self).__init__(
            # Use LSTM or GRU in the RNN
            # l0=L.LSTM(input_size, hidden_size),
            l0=L.GRU(input_size, hidden_size),
            l1=L.Linear(hidden_size, output_size)
        )

    def reset_state(self):
        self.h.reset_state()

    def __call__(self, x):
        self.v = x

        self.h = self.l0(self.v)

        self.p = F.softmax(self.l1(self.h))

        return self.p


# Generate data
# Length of the copied sequence
seqLen = 8

# Number of digits (base)
numDigits = 4

horizon = seqLen * 2

# Generate hold-out sequences for testing generalization
test = []

tx = []

for j in range(10):
    sig = []

    for i in range(seqLen):
        sig.append(np.random.randint(0, numDigits))

    test.append(sig)

    for i in range(seqLen):
        one_hot = numDigits * [0.0]
        one_hot[sig[i]] = 1.0

        tx.append(one_hot)

    for i in range(seqLen):
        one_hot = numDigits * [0.0]
        one_hot[sig[i]] = 1.0

        tx.append(one_hot)

testData = np.array(tx).astype(np.float32)

x = []

for e in range(6000):
    sig = []

    for i in range(seqLen):
        sig.append(np.random.randint(0, numDigits))

    if sig in test:
        continue

    for i in range(seqLen):
        one_hot = numDigits * [0.0]
        one_hot[sig[i]] = 1.0

        x.append(one_hot)

    for i in range(seqLen):
        one_hot = numDigits * [0.0]
        one_hot[sig[i]] = 1.0

        x.append(one_hot)

data = np.array(x).astype(np.float32)

rnn = RNN(numDigits, numDigits, 256)
optimizer = optimizers.Adam()
optimizer.setup(rnn)


def loss_rnn(t):
    return F.softmax_cross_entropy(rnn.p, t)


# Run through data
for it in range(6000):
    start = np.random.randint(0, data.shape[0] // horizon - 1) * horizon

    rnn.cleargrads()

    loss = 0

    # Go through data
    for i in range(horizon - 1):
        index = start + i

        rnn(Variable(data[index].reshape((1, numDigits))))

        if i >= seqLen - 1:
            loss += loss_rnn(Variable(np.array([np.argmax(data[index + 1])], dtype=np.int32)))

    loss.backward()
    loss.unchain_backward()

    print(loss)

    optimizer.update()

    print(it)

# Test
for j in range(testData.shape[0] // horizon):
    s0 = ""
    s1 = ""

    correct = True

    pred = np.zeros((1, numDigits)).astype(np.float32)

    for i in range(horizon):
        index = j * horizon + i

        # Go through data
        if i >= seqLen:
            m = np.argmax(pred)

            if m != np.argmax(testData[index]):
                correct = False

            rnn(Variable(pred))

            pred = rnn.p.data

            s1 += str(m)
        else:
            rnn(Variable(testData[index].reshape((1, numDigits))))

            pred = rnn.p.data

            s0 += str(np.argmax(testData[index]))

    print("Pass: " + str(correct) + " " + s0 + " " + s1)