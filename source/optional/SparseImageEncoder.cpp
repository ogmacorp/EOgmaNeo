// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "SparseImageEncoder.h"

#include "Layer.h"

#include <algorithm>
#include <fstream>

using namespace eogmaneo;

void SparseImageEncoderWorkItem::run(size_t threadIndex) {
	_pEncoder->activate(_cx, _cy, _inputChunkSize);
}

void SparseImageInhibitWorkItem::run(size_t threadIndex) {
	_pEncoder->inhibit(_cx, _cy);
}

void SparseImageEncoder::create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int chunkSize, int radius,
    unsigned long seed)
{
    std::mt19937 rng;
    rng.seed(seed);

    _inputWidth = inputWidth;
    _inputHeight = inputHeight;
    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;

    _chunkSize = chunkSize;

    _radius = radius;

    std::normal_distribution<float> weightDist(0.0f, 1.0f);

    int diam = radius * 2 + 1;

    int weightsPerUnit = diam * diam;

	int units = hiddenWidth * hiddenHeight;

    _weights.resize(units * weightsPerUnit);

    for (int w = 0; w < _weights.size(); w++) {
        _weights[w] = weightDist(rng);
    }

	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _hiddenStates.resize(chunksInX * chunksInY, 0);
}

const std::vector<int> &SparseImageEncoder::activate(const std::vector<float> &input, ComputeSystem &cs, int inputChunkSize) {
	_input = input;

    // Clear
    _hiddenActivations = std::vector<float>(_hiddenWidth * _hiddenHeight, 0.0f);
	
    {
        int chunksInX = _inputWidth / inputChunkSize;
        int chunksInY = _inputHeight / inputChunkSize;

        for (int cx = 0; cx < chunksInX; cx++)
            for (int cy = 0; cy < chunksInY; cy++) {
                std::shared_ptr<SparseImageEncoderWorkItem> item = std::make_shared<SparseImageEncoderWorkItem>();

                item->_cx = cx;
                item->_cy = cy;
                item->_inputChunkSize = inputChunkSize;
                item->_pEncoder = this;

                cs._pool.addItem(item);
            }
            
        cs._pool.wait();
    }

    {
        int chunksInX = _hiddenWidth / _chunkSize;
        int chunksInY = _hiddenHeight / _chunkSize;

        for (int cx = 0; cx < chunksInX; cx++)
            for (int cy = 0; cy < chunksInY; cy++) {
                std::shared_ptr<SparseImageInhibitWorkItem> item = std::make_shared<SparseImageInhibitWorkItem>();

                item->_cx = cx;
                item->_cy = cy;
                item->_pEncoder = this;

                cs._pool.addItem(item);
            }
            
        cs._pool.wait();
    }

    return _hiddenStates;
}

void SparseImageEncoder::activate(int cx, int cy, int inputChunkSize) {
    int chunksInX = _inputWidth / inputChunkSize;
    int chunksInY = _inputHeight / inputChunkSize;

    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

    // Projection
    float toInputX = static_cast<float>(_hiddenWidth) / static_cast<float>(chunksInX);
    float toInputY = static_cast<float>(_hiddenHeight) / static_cast<float>(chunksInY);

    int centerX = cx * toInputX;
    int centerY = cy * toInputY;

    int lowerX = centerX - _radius;
    int lowerY = centerY - _radius;

    int ci = cx + cy * chunksInX;

    for (int dx = 0; dx < inputChunkSize; dx++)
        for (int dy = 0; dy < inputChunkSize; dy++) {
            int x = cx * inputChunkSize + dx;
            int y = cy * inputChunkSize + dy;

            float input = _input[x + y * _inputWidth];

            // Sparsity optimization
            if (input != 0.0f) {
                for (int sx = 0; sx < diam; sx++)
                    for (int sy = 0; sy < diam; sy++) {
                        int index = sx + sy * diam;

                        int vx = lowerX + sx;
                        int vy = lowerY + sy;

                        if (vx >= 0 && vy >= 0 && vx < _hiddenWidth && vy < _hiddenHeight) {
                            int wi = index + weightsPerUnit * (x + y * _inputWidth);
                            int ii = vx + vy * _hiddenWidth;

                            _hiddenActivations[ii] += _weights[wi] * input;
                        }
                    }
            }
        }
}

void SparseImageEncoder::inhibit(int cx, int cy) {
	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    float maxValue = -99999.0f;
    int maxIndex = 0;

    for (int dx = 0; dx < _chunkSize; dx++)
        for (int dy = 0; dy < _chunkSize; dy++) {
            int x = cx * _chunkSize + dx;
            int y = cy * _chunkSize + dy;

            float value = _hiddenActivations[x + y * _hiddenWidth];

            if (value > maxValue) {
                maxValue = value;

                maxIndex = dx + dy * _chunkSize;
            }
        }

    _hiddenStates[cx + cy * chunksInX] = maxIndex;
}