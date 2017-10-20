// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "KMeansEncoder.h"

#include <algorithm>
#include <fstream>

using namespace eogmaneo;

void KMeansEncoderWorkItem::run(size_t threadIndex) {
	_pEncoder->activate(_cx, _cy);
}

void KMeansDecoderWorkItem::run(size_t threadIndex) {
	_pEncoder->reconstruct(_cx, _cy);
}

void KMeansLearnWorkItem::run(size_t threadIndex) {
    _pEncoder->learn(_cx, _cy, _alpha, _gamma, _minDistance);
}

void KMeansEncoder::create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int chunkSize, int radius,
    float initMinWeight, float initMaxWeight, unsigned long seed)
{
    std::mt19937 rng;
    rng.seed(seed);

    _inputWidth = inputWidth;
    _inputHeight = inputHeight;
    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;

    _chunkSize = chunkSize;

    _radius = radius;

    std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

    int diam = radius * 2 + 1;

    int weightsPerUnit = diam * diam;

	int units = hiddenWidth * hiddenHeight;

    _weights.resize(hiddenWidth * hiddenHeight * weightsPerUnit);

    for (int w = 0; w < _weights.size(); w++)
        _weights[w] = weightDist(rng);

	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _hiddenStates.resize(chunksInX * chunksInY, 0);
    _hiddenStatesPrev.resize(chunksInX * chunksInY, 0);

    _hiddenActivations.resize(hiddenWidth * hiddenHeight, 0.0f);
    _hiddenBiases.resize(hiddenWidth * hiddenHeight, 0.0f);
}

const std::vector<int> &KMeansEncoder::activate(const std::vector<float> &input, ComputeSystem &cs) {
	_input = input;
	
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<KMeansEncoderWorkItem> item = std::make_shared<KMeansEncoderWorkItem>();

			item->_cx = cx;
			item->_cy = cy;
			item->_pEncoder = this;

			cs._pool.addItem(item);
        }
		
	cs._pool.wait();

    return _hiddenStates;
}

const std::vector<float> &KMeansEncoder::reconstruct(const std::vector<int> &hiddenStates, ComputeSystem &cs) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _reconHiddenStates = hiddenStates;
	
	_recon.clear();
	_recon.assign(_inputWidth * _inputHeight, 0.0f);
	
	_count.clear();
	_count.assign(_inputWidth * _inputHeight, 0.0f);
	
    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<KMeansDecoderWorkItem> item = std::make_shared<KMeansDecoderWorkItem>();

			item->_cx = cx;
			item->_cy = cy;
			item->_pEncoder = this;

			cs._pool.addItem(item);
        }
		
	cs._pool.wait();
	
	// Rescale
	for (int i = 0; i < _recon.size(); i++)
		_recon[i] /= std::max(0.0001f, _count[i]);

    return _recon;
}

void KMeansEncoder::learn(float alpha, float gamma, float minDistance, ComputeSystem &cs) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<KMeansLearnWorkItem> item = std::make_shared<KMeansLearnWorkItem>();

            item->_cx = cx;
            item->_cy = cy;
            item->_pEncoder = this;
            item->_alpha = alpha;
            item->_gamma = gamma;
            item->_minDistance = minDistance;

            cs._pool.addItem(item);
        }

    cs._pool.wait();
}

void KMeansEncoder::activate(int cx, int cy) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

    int maxIndex = 0;
    float maxValue = -99999.0f;

    // Projection
    float toInputX = static_cast<float>(_inputWidth) / static_cast<float>(chunksInX);
    float toInputY = static_cast<float>(_inputHeight) / static_cast<float>(chunksInY);

    int centerX = cx * toInputX;
    int centerY = cy * toInputY;

    int lowerX = centerX - _radius;
    int lowerY = centerY - _radius;

    for (int dx = 0; dx < _chunkSize; dx++)
        for (int dy = 0; dy < _chunkSize; dy++) {
            int x = cx * _chunkSize + dx;
            int y = cy * _chunkSize + dy;

            // Compute value
            float value = 0.0f;

            for (int sx = 0; sx < diam; sx++)
                for (int sy = 0; sy < diam; sy++) {
                    int index = sx + sy * diam;

                    int vx = lowerX + sx;
                    int vy = lowerY + sy;

                    if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight) {
                        float delta = _weights[index + weightsPerUnit * (x + y * _hiddenWidth)] - _input[vx + vy * _inputWidth];

                        value += -delta * delta;
                    }
                }

            _hiddenActivations[x + y * _hiddenWidth] = value;

            value += _hiddenBiases[x + y * _hiddenWidth];
                
            if (value > maxValue) {
                maxValue = value;
                maxIndex = dx + dy * _chunkSize;
            }
        }

    _hiddenStatesPrev[cx + cy * chunksInX] = _hiddenStates[cx + cy * chunksInX];
	_hiddenStates[cx + cy * chunksInX] = maxIndex;
}

void KMeansEncoder::reconstruct(int cx, int cy) {
	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

    int maxIndex = 0;
    float maxValue = -99999.0f;

    // Projection
    float toInputX = static_cast<float>(_inputWidth) / static_cast<float>(chunksInX);
    float toInputY = static_cast<float>(_inputHeight) / static_cast<float>(chunksInY);

    int centerX = cx * toInputX;
    int centerY = cy * toInputY;

    int lowerX = centerX - _radius;
    int lowerY = centerY - _radius;

    // Retrieve view
	{
		int i = _reconHiddenStates[cx + cy * chunksInX];

        int dx = i % _chunkSize;
		int dy = i / _chunkSize;
		
		int x = cx * _chunkSize + dx;
		int y = cy * _chunkSize + dy;

		for (int sx = 0; sx < diam; sx++)
			for (int sy = 0; sy < diam; sy++) {
				int index = sx + sy * diam;
				
				int vx = lowerX + sx;
				int vy = lowerY + sy;

				if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight) {
					_recon[vx + vy * _inputWidth] += _weights[index + weightsPerUnit * (x + y * _hiddenWidth)];
					_count[vx + vy * _inputWidth] += 1.0f;
				}
			}
	}
}

void KMeansEncoder::learn(int cx, int cy, float alpha, float gamma, float minDistance) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

    int maxIndex = 0;
    float maxValue = -99999.0f;

    // Projection
    float toInputX = static_cast<float>(_inputWidth) / static_cast<float>(chunksInX);
    float toInputY = static_cast<float>(_inputHeight) / static_cast<float>(chunksInY);

    int centerX = cx * toInputX;
    int centerY = cy * toInputY;

    int lowerX = centerX - _radius;
    int lowerY = centerY - _radius;

    int winX = _hiddenStates[cx + cy * chunksInX] % _chunkSize;
    int winY = _hiddenStates[cx + cy * chunksInX] / _chunkSize;

    for (int dx = 0; dx < _chunkSize; dx++)
        for (int dy = 0; dy < _chunkSize; dy++) {
            int x = cx * _chunkSize + dx;
            int y = cy * _chunkSize + dy;

            _hiddenBiases[x + y * _hiddenWidth] += gamma * (1.0f / (_chunkSize * _chunkSize) - (dx == winX && dy == winY ? 1.0f : 0.0f));
        }

    {
        int x = cx * _chunkSize + winX;
        int y = cy * _chunkSize + winY;

        if (_hiddenActivations[x + y * _hiddenWidth] < -minDistance) { // && _hiddenStates[cx + cy * chunksInX] != _hiddenStatesPrev[cx + cy * chunksInX]
            // Compute value
            for (int sx = 0; sx < diam; sx++)
                for (int sy = 0; sy < diam; sy++) {
                    int index = sx + sy * diam;

                    int vx = lowerX + sx;
                    int vy = lowerY + sy;

                    if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight)
                        _weights[index + weightsPerUnit * (x + y * _hiddenWidth)] += alpha * (_input[vx + vy * _inputWidth] - _weights[index + weightsPerUnit * (x + y * _hiddenWidth)]);
                }
        }
    }
}

void KMeansEncoder::save(const std::string &fileName) {
    std::ofstream s(fileName);

    s << _inputWidth << " " << _inputHeight << " " << _hiddenWidth << " " << _hiddenHeight << " " << _chunkSize << " " << _radius << std::endl;
    
    for (int i = 0; i < _weights.size(); i++)
        s << _weights[i] << std::endl;

    for (int i = 0; i < _hiddenStates.size(); i++)
        s << _hiddenStates[i] << " " << _hiddenStatesPrev[i] << std::endl;
}

bool KMeansEncoder::load(const std::string &fileName) {
    std::ifstream s(fileName);

    if (!s.is_open())
        return false;

    s >> _inputWidth >> _inputHeight >> _hiddenWidth >> _hiddenHeight >> _chunkSize >> _radius;

    int diam = _radius * 2 + 1;

    int weightsPerUnit = diam * diam;

	int units = _hiddenWidth * _hiddenHeight;

    _weights.resize(_hiddenWidth * _hiddenHeight * weightsPerUnit);

    for (int w = 0; w < _weights.size(); w++)
        s >> _weights[w];

	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _hiddenStates.resize(chunksInX * chunksInY);
    _hiddenStatesPrev.resize(chunksInX * chunksInY);

    for (int i = 0; i < _hiddenStates.size(); i++)
        s >> _hiddenStates[i] >> _hiddenStatesPrev[i];

    _hiddenActivations.resize(_hiddenWidth * _hiddenHeight, 0.0f);
    _hiddenBiases.resize(_hiddenWidth * _hiddenHeight, 0.0f);

    return true;
}