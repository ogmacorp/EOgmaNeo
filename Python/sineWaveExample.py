# ----------------------------------------------------------------------------
#  EOgmaNeo
#  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the EOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import numpy as np
import eogmaneo
import matplotlib.pyplot as plt

cs = eogmaneo.ComputeSystem(8)

columnSize = 64

bounds = (-1.0, 1.0) # Range of value

lds = []

for i in range(9):
    ld = eogmaneo.LayerDesc()

    ld._width = 5
    ld._height = 5
    ld._columnSize = 32
    ld._forwardRadius = ld._backwardRadius = 2
    ld._lateralRadius = 2
    ld._ticksPerUpdate = 2
    ld._temporalHorizon = 2
    
    lds.append(ld)

h = eogmaneo.Hierarchy()

h.create([ (1, 1) ], [ columnSize ], [ True ], lds, 123)

# Set parameters
for i in range(len(lds)):
    l = h.getLayer(i)
    l._alpha = 0.01
    l._beta = 0.1
    
# Present the wave sequence
iters = 8000

for t in range(iters):
    index = t

    # Some function
    valueToEncode = np.sin(index * 0.02 * 2.0 * np.pi) * 0.25 + np.sin(index * 0.05 * 2.0 * np.pi + 0.2) * 0.15 + (((index * 0.01) % 1.25) * 2.0 - 1.0) * 0.2

    sdr =  [ int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (columnSize - 1) + 0.5) ]

    h.step(cs, [ sdr ], True)
    
    if t % 100 == 0:
        print(t)

# Recall
ts = []
vs = []
trgs = []

for t in range(300):
    t2 = t + iters

    index = t2

    # Some function
    valueToEncode = np.sin(index * 0.02 * 2.0 * np.pi) * 0.25 + np.sin(index * 0.05 * 2.0 * np.pi + 0.2) * 0.15 + (((index * 0.01) % 1.25) * 2.0 - 1.0) * 0.2

    h.step(cs, [ h.getPredictions(0) ], False)

    predSDR = h.getPredictions(0)[0] # First (only in this case) input layer prediction
    
    # Decode value
    value = predSDR / float(columnSize - 1) * (bounds[1] - bounds[0]) + bounds[0]

    ts.append(t)
    vs.append(value)
    trgs.append(valueToEncode)

    print(value)

plt.plot(ts, vs, ts, trgs)
plt.show()


