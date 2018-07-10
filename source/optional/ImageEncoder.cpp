// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

#include "Layer.h"

#include <algorithm>
#include <fstream>

using namespace eogmaneo;

void ImageEncoderActivateWorkItem::run(size_t threadIndex) {
	_pEncoder->activate(_cx, _cy);
}

void ImageEncoderReconstructWorkItem::run(size_t threadIndex) {
	_pEncoder->reconstruct(_cx, _cy);
}

void ImageEncoderLearnWorkItem::run(size_t threadIndex) {
    _pEncoder->learn(_cx, _cy, _alpha);
}

void ImageEncoder::create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int columnSize, int radius,
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

    std::uniform_real_distribution<float> weightDist(0.99f, 1.0f);

    int diam = _radius * 2 + 1;

    int weightsPerUnit = diam * diam;

	int units = _hiddenWidth * _hiddenHeight * _columnSize;

    _weights.resize(units * weightsPerUnit);

    for (int w = 0; w < _weights.size(); w++) {
        _weights[w] = weightDist(rng);
    }

    _hiddenStates.resize(_hiddenWidth * _hiddenHeight, 0);
}

const std::vector<int> &ImageEncoder::activate(ComputeSystem &cs, const std::vector<float> &inputs) {
	_inputs = inputs;

    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<ImageEncoderActivateWorkItem> item = std::make_shared<ImageEncoderActivateWorkItem>();

            item->_pEncoder = this;
            item->_cx = cx;
            item->_cy = cy;
            
            cs._pool.addItem(item);
        }
        
    cs._pool.wait();

    return _hiddenStates;
}

const std::vector<float> &ImageEncoder::reconstruct(ComputeSystem &cs, const std::vector<int> &hiddenStates) {
    _reconHiddenStates = hiddenStates;
	
	_recons.clear();
	_recons.assign(_inputWidth * _inputHeight, 0.0f);
	
	_counts.clear();
	_counts.assign(_inputWidth * _inputHeight, 0.0f);
	
    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<ImageEncoderReconstructWorkItem> item = std::make_shared<ImageEncoderReconstructWorkItem>();

			item->_pEncoder = this;
            item->_cx = cx;
			item->_cy = cy;
			
			cs._pool.addItem(item);
        }
		
	cs._pool.wait();
	
	// Rescale
	for (int i = 0; i < _recons.size(); i++)
        _recons[i] /= std::max(0.0001f, _counts[i]);

    return _recons;
}

void ImageEncoder::learn(ComputeSystem &cs, float alpha) {
    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<ImageEncoderLearnWorkItem> item = std::make_shared<ImageEncoderLearnWorkItem>();

            item->_pEncoder = this;
            item->_cx = cx;
            item->_cy = cy;
            item->_alpha = alpha;

            cs._pool.addItem(item);
        }

    cs._pool.wait();
}

void ImageEncoder::activate(int cx, int cy) {
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

                    value += _inputs[ii] * _weights[wi];
                }
            }

        if (value > maxValue) {
            maxValue = value;
            maxCellIndex = c;
        }
    }

	_hiddenStates[cx + cy * _hiddenWidth] = maxCellIndex;
}

void ImageEncoder::reconstruct(int cx, int cy) {
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

void ImageEncoder::learn(int cx, int cy, float alpha) {
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

                _weights[wi] += alpha * std::min(0.0f, _inputs[vx + vy * _inputWidth] - _weights[wi]);
            }
        }
}