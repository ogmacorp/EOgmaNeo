// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "GaborEncoder.h"

#include "Layer.h"

#include <algorithm>
#include <fstream>

using namespace eogmaneo;

void GaborEncoderActivateWorkItem::run(size_t threadIndex) {
	_pEncoder->activate(_cx, _cy);
}

void GaborEncoderReconstructWorkItem::run(size_t threadIndex) {
	_pEncoder->reconstruct(_cx, _cy);
}

void GaborEncoder::create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int columnSize, int radius,
    unsigned long seed, float sigma, float lam)
{
    std::mt19937 rng;
    rng.seed(seed);

    _inputWidth = inputWidth;
    _inputHeight = inputHeight;
    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;

    _columnSize = columnSize;

    _radius = radius;

    int diam = _radius * 2 + 1;

    int weightsPerUnit = diam * diam;

    _weights.resize(_columnSize * weightsPerUnit);

    const float pi2 = 6.2831f;

    // Generate filters
    std::uniform_real_distribution<float> angleDist(0.0f, pi2);

    int hDiam = (diam - 1) / 2;
    
    float invHDiam = 2.0f / (diam - 1);

    sigma = sigma / diam;
    
    for (int c = 0; c < _columnSize; c++) {
        float theta = angleDist(rng);
        float psi = angleDist(rng);

        for (int wi = 0; wi < weightsPerUnit; wi++) {
            int wx = wi % diam;
            int wy = wi / diam;

            int dx = wx - hDiam;
            int dy = wy - hDiam;

            float thetaX = dx * invHDiam * std::cos(theta) + dy * invHDiam * std::sin(theta);
            float thetaY = -dx * invHDiam * std::sin(theta) + dy * invHDiam * std::cos(theta);

            _weights[wi + c * weightsPerUnit] = std::exp(-0.5f * (thetaX * thetaX + thetaY * thetaY) / (sigma * sigma)) * std::cos(pi2 * thetaX / lam + psi);
        }
    }

    _hiddenStates.resize(_hiddenWidth * _hiddenHeight, 0);
}

const std::vector<int> &GaborEncoder::activate(ComputeSystem &cs, const std::vector<float> &inputs) {
	_inputs = inputs;

    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<GaborEncoderActivateWorkItem> item = std::make_shared<GaborEncoderActivateWorkItem>();

            item->_pEncoder = this;
            item->_cx = cx;
            item->_cy = cy;
            
            cs._pool.addItem(item);
        }
        
    cs._pool.wait();

    return _hiddenStates;
}

const std::vector<float> &GaborEncoder::reconstruct(ComputeSystem &cs, const std::vector<int> &hiddenStates) {
    _reconHiddenStates = hiddenStates;
	
	_recons.clear();
	_recons.assign(_inputWidth * _inputHeight, 0.0f);
	
	_counts.clear();
	_counts.assign(_inputWidth * _inputHeight, 0.0f);
	
    for (int cx = 0; cx < _hiddenWidth; cx++)
        for (int cy = 0; cy < _hiddenHeight; cy++) {
            std::shared_ptr<GaborEncoderReconstructWorkItem> item = std::make_shared<GaborEncoderReconstructWorkItem>();

			item->_pEncoder = this;
            item->_cx = cx;
			item->_cy = cy;
			
			cs._pool.addItem(item);
        }
		
	cs._pool.wait();
	
	// Rescale
	for (int i = 0; i < _recons.size(); i++)
        _recons[i] = std::min(1.0f, std::max(0.0f, _recons[i] / std::max(0.0001f, _counts[i])));

    return _recons;
}

void GaborEncoder::activate(int cx, int cy) {
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
        // Compute value
        float value = 0.0f;

        for (int sx = 0; sx < diam; sx++)
            for (int sy = 0; sy < diam; sy++) {
                int index = sx + sy * diam;

                int vx = lowerX + sx;
                int vy = lowerY + sy;

                if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight) {
                    int wi = index + weightsPerUnit * c;
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

void GaborEncoder::reconstruct(int cx, int cy) {
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

    for (int sx = 0; sx < diam; sx++)
        for (int sy = 0; sy < diam; sy++) {
            int index = sx + sy * diam;
            
            int vx = lowerX + sx;
            int vy = lowerY + sy;

            if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight) {
                int wi = index + weightsPerUnit * c;

                _recons[vx + vy * _inputWidth] += _weights[wi];
                _counts[vx + vy * _inputWidth] += 1.0f;
            }
        }
}