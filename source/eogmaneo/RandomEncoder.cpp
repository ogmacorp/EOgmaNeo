// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "RandomEncoder.h"

#include <algorithm>

using namespace eogmaneo;

void RandomEncoderWorkItem::run(size_t threadIndex) {
	_pEncoder->activate(_cx, _cy, _useDistanceMetric);
}

void RandomDecoderWorkItem::run(size_t threadIndex) {
	_pEncoder->reconstruct(_cx, _cy);
}

void RandomLearnWorkItem::run(size_t threadIndex) {
    _pEncoder->learn(_cx, _cy, _alpha, _gamma);
}

void RandomEncoder::create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int chunkSize, int radius,
    float initMinWeight, float initMaxWeight, unsigned long seed, bool normalize)
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

	if (normalize) {
		// Normalize
		for (int w = 0; w < units; w++) {
			float sum2 = 0.0f;

			for (int dx = -radius; dx <= radius; dx++)
				for (int dy = -radius; dy <= radius; dy++) {
					int index = (dx + radius) + (dy + radius) * diam;

					float weight = _weights[w * weightsPerUnit + index];

					sum2 += weight * weight;
				}

			// Normalize
			float scale = 1.0f / std::max(0.0001f, std::sqrt(sum2));

			for (int dx = -radius; dx <= radius; dx++)
				for (int dy = -radius; dy <= radius; dy++) {
					int index = (dx + radius) + (dy + radius) * diam;

					_weights[w * weightsPerUnit + index] *= scale;
				}
		}
	}

	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _hiddenStates.resize(chunksInX * chunksInY, 0);

    _hiddenActivations.resize(hiddenWidth * hiddenHeight, 0.0f);
    _hiddenBiases.resize(hiddenWidth * hiddenHeight, 0.0f);
}

const std::vector<int> &RandomEncoder::activate(const std::vector<float> &input, System &system, bool useDistanceMetric) {
	_input = input;
	
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<RandomEncoderWorkItem> item = std::make_shared<RandomEncoderWorkItem>();

			item->_cx = cx;
			item->_cy = cy;
			item->_pEncoder = this;
			item->_useDistanceMetric = useDistanceMetric;

			system._pool.addItem(item);
        }
		
	system._pool.wait();

    return _hiddenStates;
}

const std::vector<float> &RandomEncoder::reconstruct(const std::vector<int> &hiddenStates, System &system) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _reconHiddenStates = hiddenStates;
	
	_recon.clear();
	_recon.assign(_inputWidth * _inputHeight, 0.0f);
	
	_count.clear();
	_count.assign(_inputWidth * _inputHeight, 0.0f);
	
    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            /*std::shared_ptr<RandomDecoderWorkItem> item = std::make_shared<RandomDecoderWorkItem>();

			item->_cx = cx;
			item->_cy = cy;
			item->_pEncoder = this;

			system._pool.addItem(item);*/

            reconstruct(cx, cy);
        }
		
	//system._pool.wait();
	
	// Rescale
	for (int i = 0; i < _recon.size(); i++)
		_recon[i] /= std::max(0.0001f, _count[i]);

    return _recon;
}

void RandomEncoder::learn(float alpha, float gamma, System &system) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<RandomLearnWorkItem> item = std::make_shared<RandomLearnWorkItem>();

            item->_cx = cx;
            item->_cy = cy;
            item->_pEncoder = this;
            item->_alpha = alpha;
            item->_gamma = gamma;

            system._pool.addItem(item);
        }

    system._pool.wait();
}

void RandomEncoder::activate(int cx, int cy, bool useDistanceMetric) {
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

	if (useDistanceMetric) {
		for (int dx = 0; dx < _chunkSize; dx++)
			for (int dy = 0; dy < _chunkSize; dy++) {
				int x = cx * _chunkSize + dx;
				int y = cy * _chunkSize + dy;

				// Compute value
				float value = _hiddenBiases[x + y * _hiddenWidth];

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
					
				if (value > maxValue) {
					maxValue = value;
					maxIndex = dx + dy * _chunkSize;
				}
			}
	}
	else {
		for (int dx = 0; dx < _chunkSize; dx++)
			for (int dy = 0; dy < _chunkSize; dy++) {
				int x = cx * _chunkSize + dx;
				int y = cy * _chunkSize + dy;

				// Compute value
				float value = _hiddenBiases[x + y * _hiddenWidth];

				for (int sx = 0; sx < diam; sx++)
					for (int sy = 0; sy < diam; sy++) {
						int index = sx + sy * diam;

						int vx = lowerX + sx;
						int vy = lowerY + sy;

						if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight)
							value += _weights[index + weightsPerUnit * (x + y * _hiddenWidth)] * _input[vx + vy * _inputWidth];
					}
					
                _hiddenActivations[x + y * _hiddenWidth] = value;

				if (value > maxValue) {
					maxValue = value;
					maxIndex = dx + dy * _chunkSize;
				}
			}
	}

    int winx = maxIndex % _chunkSize;
    int winy = maxIndex / _chunkSize;

	_hiddenStates[cx + cy * chunksInX] = winx + winy * _chunkSize;
}

void RandomEncoder::reconstruct(int cx, int cy) {
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

void RandomEncoder::learn(int cx, int cy, float alpha, float gamma) {
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

    int winX = _reconHiddenStates[cx + cy * chunksInX] % _chunkSize;
    int winY = _reconHiddenStates[cx + cy * chunksInX] / _chunkSize;

    for (int dx = 0; dx < _chunkSize; dx++)
        for (int dy = 0; dy < _chunkSize; dy++) {
            int x = cx * _chunkSize + dx;
            int y = cy * _chunkSize + dy;

            _hiddenBiases[x + y * _hiddenWidth] += gamma * -_hiddenActivations[x + y * _hiddenWidth];
        }


    {
        int x = cx * _chunkSize + winX;
        int y = cy * _chunkSize + winY;

        // Compute value
        for (int sx = 0; sx < diam; sx++)
            for (int sy = 0; sy < diam; sy++) {
                int index = sx + sy * diam;

                int vx = lowerX + sx;
                int vy = lowerY + sy;

                if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight)
                    _weights[index + weightsPerUnit * (x + y * _hiddenWidth)] += alpha * (_input[vx + vy * _inputWidth] - _recon[vx + vy * _inputWidth]);
            }
    }
}