// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "RLEncoder.h"

#include <algorithm>
#include <fstream>
#include <assert.h>
#include <iostream>

using namespace eogmaneo;

void RLEncoderCrossWorkItem::run(size_t threadIndex) {
	_pEncoder->updateCross(_cx, _cy, _reward, _alpha, _gamma, _epsilon, _tau, _traceDecay, _minTrace);
}

void RLEncoderActionWorkItem::run(size_t threadIndex) {
	_pEncoder->updateAction(_cx, _cy, _reward, _alpha, _gamma, _epsilon, _tau, _traceDecay, _minTrace);
}

void RLEncoder::create(int actionWidth, int actionHeight, int actionChunkSize, int hiddenWidth, int hiddenHeight, int hiddenChunkSize, int actionRadius, int crossRadius,
    unsigned long seed)
{
    _rng.seed(seed);

    _actionWidth = actionWidth;
    _actionHeight = actionHeight;
    _actionChunkSize = actionChunkSize;
    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;
    _hiddenChunkSize = hiddenChunkSize;

    _actionRadius = actionRadius;
    _crossRadius = crossRadius;

    std::uniform_real_distribution<float> weightDist(-0.001f, 0.001f);

    int actionDiam = actionRadius * 2 + 1;
    int crossDiam = crossRadius * 2 + 1;

    int actionWeightsPerUnit = actionDiam * actionDiam;
    int crossWeightsPerUnit = crossDiam * crossDiam;

	int actionUnits = actionWidth * actionHeight;
    int hiddenUnits = hiddenWidth * hiddenHeight;

    _actionWeights.resize(actionUnits);
    _actionTraces.resize(actionUnits);

    for (int i = 0; i < actionUnits; i++) {
        _actionWeights[i].resize(actionWeightsPerUnit);

        for (int j = 0; j < actionWeightsPerUnit; j++)
            _actionWeights[i][j] = weightDist(_rng);
    }

    _crossWeights.resize(hiddenUnits);
    _crossTraces.resize(hiddenUnits);

    for (int i = 0; i < hiddenUnits; i++) {
        _crossWeights[i].resize(crossWeightsPerUnit);

        for (int j = 0; j < crossWeightsPerUnit; j++)
            _crossWeights[i][j] = weightDist(_rng);
    }

	int hiddenChunksInX = _hiddenWidth / _hiddenChunkSize;
    int hiddenChunksInY = _hiddenHeight / _hiddenChunkSize;

    _hiddenQsPrev.resize(hiddenChunksInX * hiddenChunksInY, 0.0f);
    _hiddenStates.resize(_hiddenQsPrev.size(), 0);

    int actionChunksInX = _actionWidth / _actionChunkSize;
    int actionChunksInY = _actionHeight / _actionChunkSize;

    _actionQsPrev.resize(actionChunksInX * actionChunksInY, 0.0f);
    _actions.resize(_actionQsPrev.size(), 0);

    _hiddenMaxQsPrev = _hiddenQsPrev;
    _actionMaxQsPrev = _actionQsPrev;
}

void RLEncoder::step(const std::vector<int> &predictions, ComputeSystem &cs, float reward, float alpha, float gamma, float epsilon, float tau, float traceDecay, float minTrace) {
	_predictions = predictions;

    int hiddenChunksInX = _hiddenWidth / _hiddenChunkSize;
    int hiddenChunksInY = _hiddenHeight / _hiddenChunkSize;

    assert(_predictions.size() == (hiddenChunksInX * hiddenChunksInY));

    for (int cx = 0; cx < hiddenChunksInX; cx++)
        for (int cy = 0; cy < hiddenChunksInY; cy++) {
            std::shared_ptr<RLEncoderCrossWorkItem> item = std::make_shared<RLEncoderCrossWorkItem>();

			item->_cx = cx;
			item->_cy = cy;
			item->_pEncoder = this;
            item->_reward = reward;
            item->_alpha = alpha;
            item->_gamma = gamma;
            item->_epsilon = epsilon;
            item->_tau = tau;
            item->_traceDecay = traceDecay;
            item->_minTrace = minTrace;

			cs._pool.addItem(item);
        }

    int actionChunksInX = _actionWidth / _actionChunkSize;
    int actionChunksInY = _actionHeight / _actionChunkSize;

    for (int cx = 0; cx < actionChunksInX; cx++)
        for (int cy = 0; cy < actionChunksInY; cy++) {
            std::shared_ptr<RLEncoderActionWorkItem> item = std::make_shared<RLEncoderActionWorkItem>();

			item->_cx = cx;
			item->_cy = cy;
			item->_pEncoder = this;
            item->_reward = reward;
            item->_alpha = alpha;
            item->_gamma = gamma;
            item->_epsilon = epsilon;
            item->_tau = tau;
            item->_traceDecay = traceDecay;
            item->_minTrace = minTrace;

			cs._pool.addItem(item);
        }
		
	cs._pool.wait();
}

void RLEncoder::updateCross(int cx, int cy, float reward, float alpha, float gamma, float epsilon, float tau, float traceDecay, float minTrace) {
    int hiddenChunksInX = _hiddenWidth / _hiddenChunkSize;
    int hiddenChunksInY = _hiddenHeight / _hiddenChunkSize;

    int h = cx + cy * hiddenChunksInX;

    int diam = _crossRadius * 2 + 1;
    int weightsPerUnit = diam * diam;
    int chunkRadius = std::ceil(static_cast<float>(_crossRadius) / _hiddenChunkSize);

    int centerX = (cx + 0.5f) * _hiddenChunkSize;
    int centerY = (cy + 0.5f) * _hiddenChunkSize;

    int lowerX = centerX - _crossRadius;
    int lowerY = centerY - _crossRadius;

    int upperX = centerX + _crossRadius;
    int upperY = centerY + _crossRadius;

    std::vector<float> qs(_hiddenChunkSize * _hiddenChunkSize, 0.0f);

    // Compute values
    for (int dcx = -chunkRadius; dcx <= chunkRadius; dcx++)
        for (int dcy = -chunkRadius; dcy <= chunkRadius; dcy++) {
            int ocx = cx + dcx;
            int ocy = cy + dcy;

            if (ocx >= 0 && ocx < hiddenChunksInX && ocy >= 0 && ocy < hiddenChunksInY) {
                int chunkIndex = ocx + ocy * hiddenChunksInX;

                int maxIndex = _predictions[chunkIndex];

                int dx = maxIndex % _hiddenChunkSize;
                int dy = maxIndex / _hiddenChunkSize;

                int x = ocx * _hiddenChunkSize + dx;
                int y = ocy * _hiddenChunkSize + dy;

                if (x >= lowerX && x < upperX && y >= lowerY && y < upperY) {
                    int wi = (x - lowerX) + (y - lowerY) * diam;

                    for (int c = 0; c < qs.size(); c++) {
                        int ddx = c % _hiddenChunkSize;
                        int ddy = c / _hiddenChunkSize;

                        int index = (cx * _hiddenChunkSize + ddx) + (cy * _hiddenChunkSize + ddy) * _hiddenWidth;
                        
                        qs[c] += _crossWeights[index][wi];
                    }
                }
            }
        }

    int hiddenStatePrev = _hiddenStates[h];

    int maxIndex = 0;

    for (int c = 1; c < qs.size(); c++)
        if (qs[c] > qs[maxIndex])
            maxIndex = c;

    float maxQ = qs[maxIndex];

    // Explore
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    if (dist01(_rng) < epsilon) {
        std::uniform_int_distribution<int> chunkDist(0, qs.size() - 1);

        _hiddenStates[h] = chunkDist(_rng);
    }
    else
        _hiddenStates[h] = maxIndex;

    float nextQ = qs[_hiddenStates[h]];

    float tdError = std::max(reward + gamma * maxQ - tau * (_hiddenMaxQsPrev[h] - _hiddenQsPrev[h]), reward + gamma * maxQ - tau * (maxQ - qs[hiddenStatePrev])) - _hiddenQsPrev[h];

    _hiddenQsPrev[h] = nextQ;
    _hiddenMaxQsPrev[h] = maxQ;

    // Update weights and traces
    for (int c = 0; c < qs.size(); c++) {
        int ddx = c % _hiddenChunkSize;
        int ddy = c / _hiddenChunkSize;

        int index = (cx * _hiddenChunkSize + ddx) + (cy * _hiddenChunkSize + ddy) * _hiddenWidth;

        for (std::unordered_map<int, float>::iterator it = _crossTraces[index].begin(); it != _crossTraces[index].end();) {
            _crossWeights[index][it->first] += alpha * tdError * it->second;

            it->second *= traceDecay;

            if (it->second < minTrace)
                it = _crossTraces[index].erase(it);
            else
                it++;
        }
    }

    int ddx = _hiddenStates[h] % _hiddenChunkSize;
    int ddy = _hiddenStates[h] / _hiddenChunkSize;

    int index = (cx * _hiddenChunkSize + ddx) + (cy * _hiddenChunkSize + ddy) * _hiddenWidth;

    for (int dcx = -chunkRadius; dcx <= chunkRadius; dcx++)
        for (int dcy = -chunkRadius; dcy <= chunkRadius; dcy++) {
            int ocx = cx + dcx;
            int ocy = cy + dcy;

            if (ocx >= 0 && ocx < hiddenChunksInX && ocy >= 0 && ocy < hiddenChunksInY) {
                int chunkIndex = ocx + ocy * hiddenChunksInX;

                int maxIndex = _predictions[chunkIndex];

                int dx = maxIndex % _hiddenChunkSize;
                int dy = maxIndex / _hiddenChunkSize;

                int x = ocx * _hiddenChunkSize + dx;
                int y = ocy * _hiddenChunkSize + dy;

                if (x >= lowerX && x < upperX && y >= lowerY && y < upperY) {
                    int wi = (x - lowerX) + (y - lowerY) * diam;

                    _crossTraces[index][wi] = 1.0f;
                }
            }
        }
}

void RLEncoder::updateAction(int cx, int cy, float reward, float alpha, float gamma, float epsilon, float tau, float traceDecay, float minTrace) {
    int actionChunksInX = _actionWidth / _actionChunkSize;
    int actionChunksInY = _actionHeight / _actionChunkSize;

    int hiddenChunksInX = _hiddenWidth / _hiddenChunkSize;
    int hiddenChunksInY = _hiddenHeight / _hiddenChunkSize;

    int h = cx + cy * actionChunksInX;

    int diam = _actionRadius * 2 + 1;
    int weightsPerUnit = diam * diam;
    int chunkRadius = std::ceil(static_cast<float>(_actionRadius) / _hiddenChunkSize);

    // Projection
    float toInputX = static_cast<float>(_hiddenWidth) / static_cast<float>(actionChunksInX);
    float toInputY = static_cast<float>(_hiddenHeight) / static_cast<float>(actionChunksInY);

    int centerX = (cx + 0.5f) * toInputX;
    int centerY = (cy + 0.5f) * toInputY;

    int lowerX = centerX - _actionRadius;
    int lowerY = centerY - _actionRadius;

    int upperX = centerX + _actionRadius;
    int upperY = centerY + _actionRadius;

    int chunkCenterX = std::round(static_cast<float>(centerX) / _hiddenChunkSize);
    int chunkCenterY = std::round(static_cast<float>(centerY) / _hiddenChunkSize);

    std::vector<float> qs(_actionChunkSize * _actionChunkSize, 0.0f);

    // Compute values
    for (int dcx = -chunkRadius; dcx <= chunkRadius; dcx++)
        for (int dcy = -chunkRadius; dcy <= chunkRadius; dcy++) {
            int ocx = chunkCenterX + dcx;
            int ocy = chunkCenterY + dcy;

            if (ocx >= 0 && ocx < hiddenChunksInX && ocy >= 0 && ocy < hiddenChunksInY) {
                int chunkIndex = ocx + ocy * hiddenChunksInX;

                int maxIndex = _predictions[chunkIndex];

                int dx = maxIndex % _hiddenChunkSize;
                int dy = maxIndex / _hiddenChunkSize;

                int x = ocx * _hiddenChunkSize + dx;
                int y = ocy * _hiddenChunkSize + dy;

                if (x >= lowerX && x < upperX && y >= lowerY && y < upperY) {
                    int wi = (x - lowerX) + (y - lowerY) * diam;

                    for (int c = 0; c < qs.size(); c++) {
                        int ddx = c % _actionChunkSize;
                        int ddy = c / _actionChunkSize;

                        int index = (cx * _actionChunkSize + ddx) + (cy * _actionChunkSize + ddy) * _actionWidth;

                        qs[c] += _actionWeights[index][wi];
                    }
                }
            }
        }

    int actionPrev = _actions[h];

    int maxIndex = 0;

    for (int c = 1; c < qs.size(); c++)
        if (qs[c] > qs[maxIndex])
            maxIndex = c;

    float maxQ = qs[maxIndex];

    // Explore
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    if (dist01(_rng) < epsilon) {
        std::uniform_int_distribution<int> chunkDist(0, qs.size() - 1);

        _actions[h] = chunkDist(_rng);
    }
    else
        _actions[h] = maxIndex;

    float nextQ = qs[_actions[h]];

    float tdError = std::max(reward + gamma * maxQ - tau * (_actionMaxQsPrev[h] - _actionQsPrev[h]), reward + gamma * maxQ - tau * (maxQ - qs[actionPrev])) - _actionQsPrev[h];

    _actionQsPrev[h] = nextQ;
    _actionMaxQsPrev[h] = maxQ;

    // Update weights and traces
    for (int c = 0; c < qs.size(); c++) {
        int ddx = c % _actionChunkSize;
        int ddy = c / _actionChunkSize;

        int index = (cx * _actionChunkSize + ddx) + (cy * _actionChunkSize + ddy) * _actionWidth;

        for (std::unordered_map<int, float>::iterator it = _actionTraces[index].begin(); it != _actionTraces[index].end();) {
            _actionWeights[index][it->first] += alpha * tdError * it->second;

            it->second *= traceDecay;

            if (it->second < minTrace)
                it = _actionTraces[index].erase(it);
            else
                it++;
        }
    }

    int ddx = _actions[h] % _actionChunkSize;
    int ddy = _actions[h] / _actionChunkSize;

    int index = (cx * _actionChunkSize + ddx) + (cy * _actionChunkSize + ddy) * _actionWidth;

    for (int dcx = -chunkRadius; dcx <= chunkRadius; dcx++)
        for (int dcy = -chunkRadius; dcy <= chunkRadius; dcy++) {
            int ocx = chunkCenterX + dcx;
            int ocy = chunkCenterY + dcy;

            if (ocx >= 0 && ocx < hiddenChunksInX && ocy >= 0 && ocy < hiddenChunksInY) {
                int chunkIndex = ocx + ocy * hiddenChunksInX;

                int maxIndex = _predictions[chunkIndex];

                int dx = maxIndex % _hiddenChunkSize;
                int dy = maxIndex / _hiddenChunkSize;

                int x = ocx * _hiddenChunkSize + dx;
                int y = ocy * _hiddenChunkSize + dy;

                if (x >= lowerX && x < upperX && y >= lowerY && y < upperY) {
                    int wi = (x - lowerX) + (y - lowerY) * diam;

                    _actionTraces[index][wi] = 1.0f;
                }
            }
        }
}