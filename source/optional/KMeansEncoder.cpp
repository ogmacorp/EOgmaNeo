// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "KMeansEncoder.h"

#include "Layer.h"

#include <algorithm>
#include <fstream>

using namespace eogmaneo;

void KMeansEncoderActivateWorkItem::run(size_t threadIndex) {
	_pEncoder->activate(_cx, _cy);
}

void KMeansEncoderReconstructWorkItem::run(size_t threadIndex) {
	_pEncoder->reconstruct(_cx, _cy);
}

void KMeansEncoderLearnWorkItem::run(size_t threadIndex) {
    _pEncoder->learn(_cx, _cy, _alpha);
}

void KMeansEncoder::create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int columnSize, int radius,
    float initMinWeight, float initMaxWeight,
    unsigned long seed)
{
    std::mt19937 rng;
    rng.seed(seed);

    _inputWidth = inputWidth;
    _inputHeight = inputHeight;
    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;

    _columnSize = columnSize;

    _radius = radius;

    std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

    int diam = _radius * 2 + 1;

    int weightsPerUnit = diam * diam;

	int units = _hiddenWidth * _hiddenHeight * _columnSize;

    _weights.resize(units * weightsPerUnit);

    for (int w = 0; w < _weights.size(); w++) {
        _weights[w] = weightDist(rng);
    }

    _hiddenStates.resize(_hiddenWidth * _hiddenHeight, 0);
}

const std::vector<int> &KMeansEncoder::activate(ComputeSystem &cs, const std::vector<float> &inputs) {
	_inputs = inputs;

    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<KMeansEncoderActivateWorkItem> item = std::make_shared<KMeansEncoderActivateWorkItem>();

            item->_pEncoder = this;
            item->_cx = cx;
            item->_cy = cy;
            
            cs._pool.addItem(item);
        }
        
    cs._pool.wait();

    return _hiddenStates;
}

const std::vector<float> &KMeansEncoder::reconstruct(ComputeSystem &cs, const std::vector<int> &hiddenStates) {
    _reconHiddenStates = hiddenStates;
	
	_recons.clear();
	_recons.assign(_inputWidth * _inputHeight, 0.0f);
	
	_counts.clear();
	_counts.assign(_inputWidth * _inputHeight, 0.0f);
	
    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<KMeansEncoderReconstructWorkItem> item = std::make_shared<KMeansEncoderReconstructWorkItem>();

			item->_pEncoder = this;
            item->_cx = cx;
			item->_cy = cy;
			
			cs._pool.addItem(item);
        }
		
	cs._pool.wait();
	
	// Rescale
	for (int i = 0; i < _recons.size(); i++)
        _recons[i] = _recons[i] / std::max(0.0001f, _counts[i]);

    return _recons;
}

void KMeansEncoder::learn(ComputeSystem &cs, float alpha) {
    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<KMeansEncoderLearnWorkItem> item = std::make_shared<KMeansEncoderLearnWorkItem>();

            item->_pEncoder = this;
            item->_cx = cx;
            item->_cy = cy;
            item->_alpha = alpha;

            cs._pool.addItem(item);
        }

    cs._pool.wait();
}

void KMeansEncoder::activate(int cx, int cy) {
    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

    int maxCellIndex = 0;
    float maxValue = -99999.0f;

    // Projection
    float toInputX = static_cast<float>(_inputWidth) / static_cast<float>(_hiddenWidth);
    float toInputY = static_cast<float>(_inputHeight) / static_cast<float>(_hiddenHeight);

    int centerX = cx * toInputX + 0.5f;
    int centerY = cy * toInputY + 0.5f;

    int lowerX = centerX - _radius;
    int lowerY = centerY - _radius;

    for (int c = 0; c < _columnSize; c++) {
        int ui = cx + cy * _hiddenWidth + c * _hiddenWidth * _hiddenHeight;

        // Compute value
        float value = 0.0f;

        for (int sx = 0; sx < diam; sx++)
            for (int sy = 0; sy < diam; sy++) {
                int index = sx + sy * diam;

                int vx = lowerX + sx;
                int vy = lowerY + sy;

                if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight) {
                    int wi = index + weightsPerUnit * ui;
                    int ii = vx + vy * _inputWidth;

                    float d = _inputs[ii] - _weights[wi];
                    
                    value += -d * d;
                }
            }

        if (value > maxValue) {
            maxValue = value;
            maxCellIndex = c;
        }
    }

	_hiddenStates[cx + cy * _hiddenWidth] = maxCellIndex;
}

void KMeansEncoder::reconstruct(int cx, int cy) {
    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

    int maxCellIndex = 0;
    float maxValue = -99999.0f;

    // Projection
    float toInputX = static_cast<float>(_inputWidth) / static_cast<float>(_hiddenWidth);
    float toInputY = static_cast<float>(_inputHeight) / static_cast<float>(_hiddenHeight);

    int centerX = cx * toInputX + 0.5f;
    int centerY = cy * toInputY + 0.5f;

    int lowerX = centerX - _radius;
    int lowerY = centerY - _radius;

    int c = _reconHiddenStates[cx + cy * _hiddenWidth];

    int ui = cx + cy * _hiddenWidth + c * _hiddenWidth * _hiddenHeight;

    for (int sx = 0; sx < diam; sx++)
        for (int sy = 0; sy < diam; sy++) {
            int index = sx + sy * diam;
            
            int vx = lowerX + sx;
            int vy = lowerY + sy;

            if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight) {
                int wi = index + weightsPerUnit * ui;

                _recons[vx + vy * _inputWidth] += _weights[wi];
                _counts[vx + vy * _inputWidth] += 1.0f;
            }
        }
}

void KMeansEncoder::learn(int cx, int cy, float alpha) {
    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

    int maxCellIndex = 0;
    float maxValue = -99999.0f;

    // Projection
    float toInputX = static_cast<float>(_inputWidth) / static_cast<float>(_hiddenWidth);
    float toInputY = static_cast<float>(_inputHeight) / static_cast<float>(_hiddenHeight);

    int centerX = cx * toInputX + 0.5f;
    int centerY = cy * toInputY + 0.5f;

    int lowerX = centerX - _radius;
    int lowerY = centerY - _radius;

    int c = _hiddenStates[cx + cy * _hiddenWidth];

    int ui = cx + cy * _hiddenWidth + c * _hiddenWidth * _hiddenHeight;

    // Compute value
    for (int sx = 0; sx < diam; sx++)
        for (int sy = 0; sy < diam; sy++) {
            int index = sx + sy * diam;

            int vx = lowerX + sx;
            int vy = lowerY + sy;

            if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight) {
                int wi = index + weightsPerUnit * ui;

                _weights[wi] += alpha * (_inputs[vx + vy * _inputWidth] - _weights[wi]);
            }
        }
}