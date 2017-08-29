// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ImageEncoder.h"

#include <algorithm>
#include <fstream>

using namespace eogmaneo;

void ImageEncoderWorkItem::run(size_t threadIndex) {
	_pEncoder->activate(_cx, _cy);
}

void ImageDecoderWorkItem::run(size_t threadIndex) {
	_pEncoder->reconstruct(_cx, _cy);
}

void ImageLearnWorkItem::run(size_t threadIndex) {
    _pEncoder->learn(_cx, _cy, _alpha);
}

void ImageEncoder::create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int chunkSize, int radius,
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

    std::uniform_real_distribution<float> weightDist(0.9999f, 1.0f);

    int diam = radius * 2 + 1;

    int weightsPerUnit = diam * diam;

	int units = hiddenWidth * hiddenHeight;

    _weights0.resize(hiddenWidth * hiddenHeight * weightsPerUnit);
    _weights1.resize(hiddenWidth * hiddenHeight * weightsPerUnit);

    for (int w = 0; w < _weights0.size(); w++) {
        _weights0[w] = weightDist(rng);
        _weights1[w] = weightDist(rng);
    }

	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _hiddenStates.resize(chunksInX * chunksInY, 0);

    _hiddenActivations.resize(hiddenWidth * hiddenHeight, 0.0f);
}

const std::vector<int> &ImageEncoder::activate(const std::vector<float> &input, ComputeSystem &cs) {
	_input = input;
	
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<ImageEncoderWorkItem> item = std::make_shared<ImageEncoderWorkItem>();

			item->_cx = cx;
			item->_cy = cy;
			item->_pEncoder = this;

			cs._pool.addItem(item);
        }
		
	cs._pool.wait();

    return _hiddenStates;
}

const std::vector<float> &ImageEncoder::reconstruct(const std::vector<int> &hiddenStates, ComputeSystem &cs) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _reconHiddenStates = hiddenStates;
	
	_recon.clear();
	_recon.assign(_inputWidth * _inputHeight, 0.0f);
	
	_count.clear();
	_count.assign(_inputWidth * _inputHeight, 0.0f);
	
    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<ImageDecoderWorkItem> item = std::make_shared<ImageDecoderWorkItem>();

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

void ImageEncoder::learn(float alpha, ComputeSystem &cs) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<ImageLearnWorkItem> item = std::make_shared<ImageLearnWorkItem>();

            item->_cx = cx;
            item->_cy = cy;
            item->_pEncoder = this;
            item->_alpha = alpha;

            cs._pool.addItem(item);
        }

    cs._pool.wait();
}

void ImageEncoder::activate(int cx, int cy) {
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
                        int wi = index + weightsPerUnit * (x + y * _hiddenWidth);
                        int ii = vx + vy * _inputWidth;

                        value += _weights0[wi] * _input[ii];// + _weights1[wi] * (1.0f - _input[ii]);
                    }
                }
                
            _hiddenActivations[x + y * _hiddenWidth] = value;

            if (value > maxValue) {
                maxValue = value;
                maxIndex = dx + dy * _chunkSize;
            }
        }

	_hiddenStates[cx + cy * chunksInX] = maxIndex;
}

void ImageEncoder::reconstruct(int cx, int cy) {
	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

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
					_recon[vx + vy * _inputWidth] += _weights0[index + weightsPerUnit * (x + y * _hiddenWidth)];// + (1.0f - _weights1[index + weightsPerUnit * (x + y * _hiddenWidth)]);
					_count[vx + vy * _inputWidth] += 1.0f; // 2.0f
				}
			}
	}
}

void ImageEncoder::learn(int cx, int cy, float alpha) {
    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    int diam = _radius * 2 + 1;
    int weightsPerUnit = diam * diam;

    // Projection
    float toInputX = static_cast<float>(_inputWidth) / static_cast<float>(chunksInX);
    float toInputY = static_cast<float>(_inputHeight) / static_cast<float>(chunksInY);

    int centerX = cx * toInputX;
    int centerY = cy * toInputY;

    int lowerX = centerX - _radius;
    int lowerY = centerY - _radius;

    int winX = _hiddenStates[cx + cy * chunksInX] % _chunkSize;
    int winY = _hiddenStates[cx + cy * chunksInX] / _chunkSize;

    int x = cx * _chunkSize + winX;
    int y = cy * _chunkSize + winY;

    // Compute value
    for (int sx = 0; sx < diam; sx++)
        for (int sy = 0; sy < diam; sy++) {
            int index = sx + sy * diam;

            int vx = lowerX + sx;
            int vy = lowerY + sy;

            if (vx >= 0 && vy >= 0 && vx < _inputWidth && vy < _inputHeight) {
                _weights0[index + weightsPerUnit * (x + y * _hiddenWidth)] += alpha * (_input[vx + vy * _inputWidth] - _weights0[index + weightsPerUnit * (x + y * _hiddenWidth)]);

                //_weights1[index + weightsPerUnit * (x + y * _hiddenWidth)] += alpha * (1.0f - _input[vx + vy * _inputWidth] - _weights1[index + weightsPerUnit * (x + y * _hiddenWidth)]);
            }
        }
}

void ImageEncoder::save(const std::string &fileName) {
    std::ofstream s(fileName);

    s << _inputWidth << " " << _inputHeight << " " << _hiddenWidth << " " << _hiddenHeight << " " << _chunkSize << " " << _radius << std::endl;

    for (int w = 0; w < _weights0.size(); w++)
        s << _weights0[w] << " " << _weights1[w] << std::endl;
}

bool ImageEncoder::load(const std::string &fileName) {
    std::ifstream s(fileName);

    if (!s.is_open())
        return false;

    s >> _inputWidth >> _inputHeight >> _hiddenWidth >> _hiddenHeight >> _chunkSize >> _radius;

    int diam = _radius * 2 + 1;

    int weightsPerUnit = diam * diam;

	int units = _hiddenWidth * _hiddenHeight;

    _weights0.resize(_hiddenWidth * _hiddenHeight * weightsPerUnit);
    _weights1.resize(_hiddenWidth * _hiddenHeight * weightsPerUnit);

    for (int w = 0; w < _weights0.size(); w++)
        s >> _weights0[w] >> _weights1[w];

	int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    _hiddenStates.resize(chunksInX * chunksInY, 0);

    _hiddenActivations.resize(_hiddenWidth * _hiddenHeight, 0.0f);

    return true;
}