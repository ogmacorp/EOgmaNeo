// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Position2DEncoder.h"

#include <algorithm>

using namespace eogmaneo;

void Position2DEncoder::create(int chunkSize, float scale, unsigned long seed) {
    _chunkSize = chunkSize;
    _scale = scale;
    _seed = seed;
}

int Position2DEncoder::activate(const std::pair<float, float> &position, ComputeSystem &cs) {
	int maxIndex = 0;
    float maxValue = -99999.0f;

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    int xi = std::round(std::get<0>(position) * _scale);
    int yi = std::round(std::get<1>(position) * _scale);

    for (int dx = 0; dx < _chunkSize; dx++)
        for (int dy = 0; dy < _chunkSize; dy++) {
            int xt = xi + dx;
            int yt = yi + dy;
            
            std::mt19937 rng(xt + yt * 1000 + _seed);

            float value = dist01(rng);

            if (value > maxValue) {
                maxValue = value;
                maxIndex = dx + dy * _chunkSize;
            }
        }

    return maxIndex;
}