# ----------------------------------------------------------------------------
#  EOgmaNeo
#  Copyright(c) 2018 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of EOgmaNeo is licensed to you under the terms described
#  in the EOGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# -*- coding: utf-8 -*-

import numpy as np
import eogmaneo

cs = eogmaneo.ComputeSystem(4)

# Length of the copied sequence
seqLen = 8

# Number of digits (base)
numDigits = 4

# Generate hold-out sequences for testing generalization
testSequences = []

for i in range(10):
    sig = []

    for j in range(seqLen):
        sig.append(np.random.randint(0, numDigits))

    testSequences.append(sig)

# Create hierarchy
lds = []

for i in range(6):
    ld = eogmaneo.LayerDesc()

    ld._width = 20
    ld._height = 20
    ld._forwardRadius = ld._backwardRadius = 10
    ld._chunkSize = 4
    ld._alpha = 0.4
    ld._beta = 0.4
    ld._ticksPerUpdate = 2
    ld._temporalHorizon = 4
    lds.append(ld)

h = eogmaneo.Hierarchy()

# Direct encoding
chunkSize = int(np.ceil(np.sqrt(float(numDigits))))

h.create([(chunkSize, chunkSize)], [chunkSize], [True], lds, 123)

# Track running average of error
avgError = 1.0

# Train
for e in range(6000):
    sig = []

    sigStr = ""

    for i in range(seqLen):
        sig.append(np.random.randint(0,
                                     numDigits))  # Maximum number of symbols is 4x4=16. 0 is the delimiter. May use less here (using 9 for visualization alignment)

        sigStr += str(sig[len(sig) - 1])

    # Skip hold-out sequences
    if sig in testSequences:
        continue

    for i in range(seqLen):
        h.step([[sig[i]]], cs, False)

    # Recall
    allCorrect = True

    predStr = ""

    for i in range(seqLen):
        pred = h.getPredictions(0)[0]

        predStr += str(pred)

        if pred != sig[i]:
            allCorrect = False

        h.step([[sig[i]]], cs, True)

    # Decay error
    avgError = 0.99 * avgError + 0.01 * (1.0 - float(allCorrect))

    print("Pass: " + str(allCorrect) + " " + sigStr + " " + predStr + " E: " + str(avgError))

# Test on hold out sequences
print("=================== Hold Out ===================")

for i in range(len(testSequences)):
    sig = testSequences[i]

    sigStr = ""

    for i in range(seqLen):
        sigStr += str(sig[i])

    for i in range(seqLen):
        h.step([[sig[i]]], cs, False)

    # Recall
    allCorrect = True

    predStr = ""

    for i in range(seqLen):
        pred = h.getPredictions(0)[0]

        predStr += str(pred)

        if pred != sig[i]:
            allCorrect = False

        h.step([[sig[i]]], cs, False)

    print("Pass: " + str(allCorrect) + " " + sigStr + " " + predStr)