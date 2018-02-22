# ----------------------------------------------------------------------------
#  EOgmaNeo
#  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the EOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import numpy as np
import eogmaneo

system = eogmaneo.ComputeSystem(4)

chunkSize = 8

unitsPerChunk = chunkSize * chunkSize

bounds = (-1.0, 1.0) # Range of value

lds = []

for i in range(3):
    ld = eogmaneo.LayerDesc()

    ld._width = 16
    ld._height = 16
    ld._chunkSize = 8
    ld._forwardRadius = ld._backwardRadius = 12
    ld._alpha = 0.4
    ld._beta = 0.4
    ld._temporalHorizon = 2

    lds.append(ld)

h = eogmaneo.Hierarchy()

h.create([ ( chunkSize, chunkSize) ], [ chunkSize ], [ True ], lds, 123)

# Present the sine wave sequence for 1000 steps
for t in range(5000):
    valueToEncode = np.sin(t * 0.02 * 2.0 * np.pi) # Test value

    # Single-chunk SDR
    chunkedSDR = [ int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (unitsPerChunk - 1) + 0.5) ]

    h.step([ chunkedSDR ], system, True)

# Recall
for t in range(100):
    predSDR = h.getPredictions(0) # First (only in this case) input layer prediction

    # Decode value
    value = predSDR[0] / float(unitsPerChunk - 1) * (bounds[1] - bounds[0]) + bounds[0]

    print(value)

    h.step([ predSDR ], system, False)

h.save("sineSave.eohr")


