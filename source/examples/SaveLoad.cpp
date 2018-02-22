// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>

#include <Hierarchy.h>
using namespace eogmaneo;


// define range function (only once)
template <typename T>
std::vector <T> range(T N1, T N2) {
    std::vector<T> numbers(N2 - N1);
    iota(numbers.begin(), numbers.end(), N1);
    return numbers;
}


int main() {
    const int hiddenWidth = 16;
    const int hiddenHeight = 16;

    const int numLayers = 3;
    const int layerSize = 16;
    const int chunkSize = 8;

    const int unitsPerChunk = chunkSize * chunkSize;

    const std::vector<float> bounds = { -1.0f, 1.0f };

    auto system = std::make_shared<ComputeSystem>(4);

    std::vector<LayerDesc> lds;

    for (int l = 0; l < numLayers; l++)
    {
        LayerDesc ld;
        ld._width = layerSize;
        ld._height = layerSize;
        ld._chunkSize = chunkSize;
        ld._forwardRadius = 12;
        ld._backwardRadius = 12;
        ld._alpha = 0.4f;
        ld._beta = 0.4f;
        ld._temporalHorizon = 2;

        lds.push_back(ld);
    }

    std::vector<std::pair<int, int>> inputSizes;
    inputSizes.push_back(std::pair<int, int>{chunkSize, chunkSize});

    std::vector<int> inputChunkSizes;
    inputChunkSizes.push_back(chunkSize);

    std::vector<bool> predictInputs;
    predictInputs.push_back(true);

    Hierarchy h1;
    h1.create(inputSizes, inputChunkSizes, predictInputs, lds, 123);

    Hierarchy h2;
    h2.create(inputSizes, inputChunkSizes, predictInputs, lds, 123);

    // Present a sine wave sequence
    std::vector<int> arr = range(0, 5000);
    for (auto t : arr) {
        float valueToEncode = sinf(t * 0.02f * 2.0f * M_PI);

        std::vector<std::vector<int>> chunkedSDR(1);
        chunkedSDR[0].push_back(int((valueToEncode - bounds[0]) / (bounds[1] - bounds[0]) * (unitsPerChunk - 1) + 0.5f));

        h1.step(chunkedSDR, *system, true);
    }

    h1.save("sineSave.eohr");
    h2.load("sineSave.eohr");

    std::vector<float> results1;
    std::vector<float> results2;

    // Recall the sequence
    arr = range(0, 100);
    for (auto t : arr) {
        // First input layer prediction
        const std::vector<int> predSDR = h1.getPredictions(0);

        // Decode value
        float value = predSDR[0] / float(unitsPerChunk - 1) * (bounds[1] - bounds[0]) + bounds[0];
        results1.push_back(value);

        std::vector<std::vector<int>> valueToEncode(1);
        valueToEncode[0].push_back(value);

        h1.step(valueToEncode, *system, false);
    }

    // Recall the sequence
    for (auto t : arr) {
        // First input layer prediction
        const std::vector<int> predSDR = h2.getPredictions(0);

        // Decode value
        float value = predSDR[0] / float(unitsPerChunk - 1) * (bounds[1] - bounds[0]) + bounds[0];
        results2.push_back(value);

        std::vector<std::vector<int>> valueToEncode(1);
        valueToEncode[0].push_back(value);

        h2.step(valueToEncode, *system, false);
    }

    return 0;
}
