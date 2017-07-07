// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "CornerEncoder.h"

#include <algorithm>
#include <iostream>

using namespace eogmaneo;

void CornerEncoderWorkItem::run(size_t threadIndex) {
    _pEncoder->activate(_cx, _cy);
}

void CornerEncoder::create(int inputWidth, int inputHeight, int chunkSize, int k) {
    _inputWidth = inputWidth;
    _inputHeight = inputHeight;

    _chunkSize = chunkSize;
    _k = k;

    int chunksInX = inputWidth / chunkSize;
    int chunksInY = inputHeight / chunkSize;

    _hiddenStates.resize(k);
    
    for (int order = 0; order < k; order++)
        _hiddenStates[order].resize(chunksInX * chunksInY, 0);

    _hiddenScores.resize(inputWidth * inputHeight, 0);
}

void CornerEncoder::activate(const std::vector<float> &input, System &system, float radius, float thresh, int samples) {
    _input = input;

    _radius = radius;

    _thresh = thresh;
    _samples = samples;

    int chunksInX = _inputWidth / _chunkSize;
    int chunksInY = _inputHeight / _chunkSize;

    // Pre-compute deltas
    const float pi = 3.141596f;

    float inc = 2.0f * pi / _samples;

    _deltas.resize(_samples);

    for (float i = 0; i < _samples; i++) {
        float f = i * inc;

        float xf = std::cos(f) * _radius;
        float yf = std::sin(f) * _radius;

        int dx = std::round(xf);
        int dy = std::round(yf);

        _deltas[i] = std::make_pair(dx, dy);
    }

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<CornerEncoderWorkItem> item = std::make_shared<CornerEncoderWorkItem>();

            item->_cx = cx;
            item->_cy = cy;
            item->_pEncoder = this;

            system._pool.addItem(item);
        }

    system._pool.wait();
}

bool cmp(const std::pair<int, int> &left, const std::pair<int, int> &right) {
    return std::get<0>(left) < std::get<0>(right);
}

void CornerEncoder::activate(int cx, int cy) {
    int chunksInX = _inputWidth / _chunkSize;
    int chunksInY = _inputHeight / _chunkSize;

    std::vector<std::pair<int, int>> heap;

    // Go through image in this chunk
    for (int rx = 0; rx < _chunkSize; rx++)
        for (int ry = 0; ry < _chunkSize; ry++) {
            int px = cx * _chunkSize + rx;
            int py = cy * _chunkSize + ry;

            float vc = _input[px + py * _inputWidth];

            int maxContiguous = 0;

            int contiguous = 0;

            bool prevBrighter = false;
            bool prevDarker = false;

            for (int s = 0; s < _samples; s++) {
                std::pair<int, int> delta = _deltas[s];

                int x = px + std::get<0>(delta);
                int y = py + std::get<1>(delta);

                if (x >= 0 && x < _inputWidth && y >= 0 && y < _inputHeight) {
                    float v = _input[x + y * _inputWidth];

                    if (s == 0) {
                        prevBrighter = v > vc + _thresh;
                        prevDarker = v < vc - _thresh;
                    }

                    bool brighter = v > vc + _thresh;
                    bool darker = v < vc - _thresh;

                    if ((brighter && prevBrighter) || (darker && prevDarker))
                        contiguous++;
                    else {
                        maxContiguous = std::max(maxContiguous, contiguous);
                        contiguous = 0;
                    }     

                    prevBrighter = brighter;
                    prevDarker = darker;
                }
            }

            maxContiguous = std::max(maxContiguous, contiguous);

            _hiddenScores[px + py * _inputWidth] = maxContiguous;

            heap.push_back(std::make_pair(maxContiguous, rx + ry * _chunkSize));
        }

    std::make_heap(heap.begin(), heap.end(), cmp);

    // Find max K scores
    for (int order = 0; order < _k; order++) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        
        int maxCoord = std::get<1>(heap.back());

        heap.pop_back();

        _hiddenStates[order][cx + cy * chunksInX] = maxCoord;
    }
}