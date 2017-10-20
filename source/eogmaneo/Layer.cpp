// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Layer.h"

#include <algorithm>
#include <iostream>

#include <assert.h>

using namespace eogmaneo;

float eogmaneo::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void ForwardWorkItem::run(size_t threadIndex) {
    assert(_pLayer != nullptr);

    int hiddenChunkSize = _pLayer->_chunkSize;

    int hiddenChunksInX = _pLayer->_hiddenWidth / hiddenChunkSize;
    int hiddenChunksInY = _pLayer->_hiddenHeight / hiddenChunkSize;

    int hiddenChunkX = _hiddenChunkIndex % hiddenChunksInX;
    int hiddenChunkY = _hiddenChunkIndex / hiddenChunksInX;

    // Extract input views
    int hiddenStatePrev = _pLayer->_hiddenStates[_hiddenChunkIndex];

    int dhxPrev = hiddenStatePrev % hiddenChunkSize;
    int dhyPrev = hiddenStatePrev / hiddenChunkSize;

    int hIndexPrev = (hiddenChunkX * hiddenChunkSize + dhxPrev) + (hiddenChunkY * hiddenChunkSize + dhyPrev) * _pLayer->_hiddenWidth;

    std::vector<float> chunkPredictions(hiddenChunkSize * hiddenChunkSize, 0.0f);

    for (int v = 0; v < _pLayer->_visibleLayerDescs.size(); v++) {
        if (!_pLayer->_visibleLayerDescs[v]._predict)
            continue;

        int visibleChunkSize = _pLayer->_visibleLayerDescs[v]._chunkSize;

        int visibleChunksInX = _pLayer->_visibleLayerDescs[v]._width / visibleChunkSize;
        int visibleChunksInY = _pLayer->_visibleLayerDescs[v]._height / visibleChunkSize;

        float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
        float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

        int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
        int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

        int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
        int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

        int spatialVisibleRadius = _pLayer->_visibleLayerDescs[v]._radius;

        int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

        int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

        int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
        int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

        int upperVisibleX = visibleCenterX + spatialVisibleRadius;
        int upperVisibleY = visibleCenterY + spatialVisibleRadius;

        for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
            for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                int cx = visibleChunkCenterX + dcx;
                int cy = visibleChunkCenterY + dcy;

                if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                    int visibleChunkIndex = cx + cy * visibleChunksInX;

                    int maxIndex = _pLayer->_inputs[v][visibleChunkIndex];

                    int mdx = maxIndex % visibleChunkSize;
                    int mdy = maxIndex / visibleChunkSize;

                    int vx = cx * visibleChunkSize + mdx;
                    int vy = cy * visibleChunkSize + mdy;

                    if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                        for (int c = 0; c < chunkPredictions.size(); c++) {
                            int dhx = c % hiddenChunkSize;
                            int dhy = c / hiddenChunkSize;

                            int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * _pLayer->_hiddenWidth;

                            int i = v + _pLayer->_visibleLayerDescs.size() * hIndex;

                            int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                            chunkPredictions[c] += _pLayer->_predictionWeights.front()[i][wi];
                        }
                    }
                }
            }
    }

	// Find max element
	int predHiddenIndex = 0;

	for (int c = 1; c < chunkPredictions.size(); c++) {
		if (chunkPredictions[c] > chunkPredictions[predHiddenIndex])
			predHiddenIndex = c;
	}

    float learn = hiddenStatePrev != predHiddenIndex ? _pLayer->_alpha : 0.0f;

    std::vector<float> chunkActivations(hiddenChunkSize * hiddenChunkSize, 0.0f);

    for (int v = 0; v < _pLayer->_visibleLayerDescs.size(); v++) {
        int visibleChunkSize = _pLayer->_visibleLayerDescs[v]._chunkSize;

        int visibleChunksInX = _pLayer->_visibleLayerDescs[v]._width / visibleChunkSize;
        int visibleChunksInY = _pLayer->_visibleLayerDescs[v]._height / visibleChunkSize;

        float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
        float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

        int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
        int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

        int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
        int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

        int spatialVisibleRadius = _pLayer->_visibleLayerDescs[v]._radius;

        int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

        int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

        int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
        int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

        int upperVisibleX = visibleCenterX + spatialVisibleRadius;
        int upperVisibleY = visibleCenterY + spatialVisibleRadius;

        int iPrev = v + _pLayer->_visibleLayerDescs.size() * hIndexPrev;

        for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
            for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                int cx = visibleChunkCenterX + dcx;
                int cy = visibleChunkCenterY + dcy;

                if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                    int visibleChunkIndex = cx + cy * visibleChunksInX;

                    int maxIndex = _pLayer->_inputs[v][visibleChunkIndex];
                    int maxIndexPrev = _pLayer->_inputsPrev[v][visibleChunkIndex];

                    for (int dvx = 0; dvx < visibleChunkSize; dvx++)
                        for (int dvy = 0; dvy < visibleChunkSize; dvy++) {
                            int index = dvx + dvy * visibleChunkSize;

                            int ovx = cx * visibleChunkSize + dvx;
                            int ovy = cy * visibleChunkSize + dvy;

                            if (ovx >= lowerVisibleX && ovx <= upperVisibleX && ovy >= lowerVisibleY && ovy <= upperVisibleY) {
                                int wi = (ovx - lowerVisibleX) + (ovy - lowerVisibleY) * spatialVisibleDiam;

                                int vIndex = ovx + ovy * _pLayer->_visibleLayerDescs[v]._width;
                                
                                float target = index == maxIndexPrev ? 1.0f : 0.0f;

                                float recon = std::min(1.0f, 1.0f + std::tanh(_pLayer->_reconActivationsPrev[v][vIndex] / std::max(1.0f, _pLayer->_reconCountsPrev[v][vIndex])));

                                _pLayer->_feedForwardWeights[iPrev][wi] += learn * (target - recon);
                            }
                        }

                    int mdx = maxIndex % visibleChunkSize;
                    int mdy = maxIndex / visibleChunkSize;

                    int vx = cx * visibleChunkSize + mdx;
                    int vy = cy * visibleChunkSize + mdy;

                    if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                        for (int c = 0; c < chunkActivations.size(); c++) {
                            int dhx = c % hiddenChunkSize;
                            int dhy = c / hiddenChunkSize;

                            int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * _pLayer->_hiddenWidth;

                            int i = v + _pLayer->_visibleLayerDescs.size() * hIndex;

                            int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                            chunkActivations[c] += _pLayer->_feedForwardWeights[i][wi];
                        }
                    }
                }
            }
    }

	// Find max element
	int maxHiddenIndex = 0;

	for (int c = 1; c < chunkActivations.size(); c++) {
		if (chunkActivations[c] > chunkActivations[maxHiddenIndex])
			maxHiddenIndex = c;
	}

    _pLayer->_hiddenStates[_hiddenChunkIndex] = maxHiddenIndex;

    // Reconstruct
    int dhx = maxHiddenIndex % hiddenChunkSize;
    int dhy = maxHiddenIndex / hiddenChunkSize;

    int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * _pLayer->_hiddenWidth;

    for (int v = 0; v < _pLayer->_visibleLayerDescs.size(); v++) {
        int visibleChunkSize = _pLayer->_visibleLayerDescs[v]._chunkSize;

        int visibleChunksInX = _pLayer->_visibleLayerDescs[v]._width / visibleChunkSize;
        int visibleChunksInY = _pLayer->_visibleLayerDescs[v]._height / visibleChunkSize;

        float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
        float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

        int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
        int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

        int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
        int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

        int spatialVisibleRadius = _pLayer->_visibleLayerDescs[v]._radius;

        int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

        int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

        int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
        int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

        int upperVisibleX = visibleCenterX + spatialVisibleRadius;
        int upperVisibleY = visibleCenterY + spatialVisibleRadius;

        int i = v + _pLayer->_visibleLayerDescs.size() * hIndex;

        for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
            for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                int cx = visibleChunkCenterX + dcx;
                int cy = visibleChunkCenterY + dcy;

                if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                    int visibleChunkIndex = cx + cy * visibleChunksInX;

                    for (int dvx = 0; dvx < visibleChunkSize; dvx++)
                        for (int dvy = 0; dvy < visibleChunkSize; dvy++) {
                            int index = dvx + dvy * visibleChunkSize;

                            int ovx = cx * visibleChunkSize + dvx;
                            int ovy = cy * visibleChunkSize + dvy;

                            if (ovx >= lowerVisibleX && ovx <= upperVisibleX && ovy >= lowerVisibleY && ovy <= upperVisibleY) {
                                int wi = (ovx - lowerVisibleX) + (ovy - lowerVisibleY) * spatialVisibleDiam;

                                int vIndex = ovx + ovy * _pLayer->_visibleLayerDescs[v]._width;

                                // Reconstruction
                                _pLayer->_reconActivations[v][vIndex] += _pLayer->_feedForwardWeights[i][wi];
                                _pLayer->_reconCounts[v][vIndex] += 1.0f;
                            }
                        }
                }
            }
    }
}

void BackwardWorkItem::run(size_t threadIndex) {
    assert(_pLayer != nullptr);

    int hiddenChunkSize = _pLayer->_chunkSize;

    int hiddenChunksInX = _pLayer->_hiddenWidth / hiddenChunkSize;
    int hiddenChunksInY = _pLayer->_hiddenHeight / hiddenChunkSize;

    int hiddenChunkX = _hiddenChunkIndex % hiddenChunksInX;
    int hiddenChunkY = _hiddenChunkIndex / hiddenChunksInX;

    for (int f = 0; f < _pLayer->_feedBack.size(); f++) {
        int dhx = _pLayer->_feedBack[f][_hiddenChunkIndex] % hiddenChunkSize;
        int dhy = _pLayer->_feedBack[f][_hiddenChunkIndex] / hiddenChunkSize;

        int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * _pLayer->_hiddenWidth;

        int dhxPrev = _pLayer->_feedBackPrev[f][_hiddenChunkIndex] % hiddenChunkSize;
        int dhyPrev = _pLayer->_feedBackPrev[f][_hiddenChunkIndex] / hiddenChunkSize;

        int hIndexPrev = (hiddenChunkX * hiddenChunkSize + dhxPrev) + (hiddenChunkY * hiddenChunkSize + dhyPrev) * _pLayer->_hiddenWidth;

        for (int v = 0; v < _pLayer->_visibleLayerDescs.size(); v++) {
            if (!_pLayer->_visibleLayerDescs[v]._predict)
                continue;

            int visibleChunkSize = _pLayer->_visibleLayerDescs[v]._chunkSize;

            int visibleChunksInX = _pLayer->_visibleLayerDescs[v]._width / visibleChunkSize;
            int visibleChunksInY = _pLayer->_visibleLayerDescs[v]._height / visibleChunkSize;

            float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
            float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

            int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
            int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

            int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
            int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

            int spatialVisibleRadius = _pLayer->_visibleLayerDescs[v]._radius;

            int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

            int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

            int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
            int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

            int upperVisibleX = visibleCenterX + spatialVisibleRadius;
            int upperVisibleY = visibleCenterY + spatialVisibleRadius;

            int i = v + _pLayer->_visibleLayerDescs.size() * hIndex;
            int iPrev = v + _pLayer->_visibleLayerDescs.size() * hIndexPrev;

            for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
                for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                    int cx = visibleChunkCenterX + dcx;
                    int cy = visibleChunkCenterY + dcy;

                    if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                        int visibleChunkIndex = cx + cy * visibleChunksInX;

                        int maxIndex = _pLayer->_inputs[v][visibleChunkIndex];

                        for (int dvx = 0; dvx < visibleChunkSize; dvx++)
                            for (int dvy = 0; dvy < visibleChunkSize; dvy++) {
                                int index = dvx + dvy * visibleChunkSize;

                                int ovx = cx * visibleChunkSize + dvx;
                                int ovy = cy * visibleChunkSize + dvy;

                                if (ovx >= lowerVisibleX && ovx <= upperVisibleX && ovy >= lowerVisibleY && ovy <= upperVisibleY) {
                                    int wi = (ovx - lowerVisibleX) + (ovy - lowerVisibleY) * spatialVisibleDiam;
            
                                    int vIndex = ovx + ovy * _pLayer->_visibleLayerDescs[v]._width;

                                    float target = index == maxIndex ? 1.0f : 0.0f;

                                    float recon = std::min(1.0f, 1.0f + std::tanh(_pLayer->_predictionActivationsPrev[v][vIndex]));

                                    _pLayer->_predictionWeights[f][iPrev][wi] += _pLayer->_beta * (target - recon);

                                    // Reconstruction
                                    _pLayer->_predictionActivations[v][vIndex] += _pLayer->_predictionWeights[f][i][wi];
                                    _pLayer->_predictionCounts[v][vIndex] += 1.0f;
                                }
                            }
                    }
                }
        }
    }
}

void PredictionWorkItem::run(size_t threadIndex) {
    assert(_pLayer != nullptr);

    int v = _visibleLayerIndex;

    int visibleWidth = _pLayer->_visibleLayerDescs[v]._width;
    int visibleHeight = _pLayer->_visibleLayerDescs[v]._height;

    int visibleChunkSize = _pLayer->_visibleLayerDescs[v]._chunkSize;

    int visibleChunksInX = visibleWidth / visibleChunkSize;
    int visibleChunksInY = visibleHeight / visibleChunkSize;

    int visibleChunkX = _visibleChunkIndex % visibleChunksInX;
    int visibleChunkY = _visibleChunkIndex / visibleChunksInX;

    int visibleBitsPerChunk = visibleChunkSize * visibleChunkSize;

    float maxValue = -99999.0f;
    int maxIndex = 0;

    for (int c = 0; c < visibleBitsPerChunk; c++) {
        int dvx = c % visibleChunkSize;
        int dvy = c / visibleChunkSize;

        int vIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

        _pLayer->_predictionActivations[v][vIndex] /= std::max(1.0f, _pLayer->_predictionCounts[v][vIndex]);

        if (_pLayer->_predictionActivations[v][vIndex] > maxValue) {
            maxValue = _pLayer->_predictionActivations[v][vIndex];
            maxIndex = c;
        }
    }

    _pLayer->_predictions[v][_visibleChunkIndex] = maxIndex;
}

void Layer::create(int hiddenWidth, int hiddenHeight, int chunkSize, int numFeedBack, const std::vector<VisibleLayerDesc> &visibleLayerDescs, unsigned long seed) {
    std::mt19937 rng(seed);

    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;
    _chunkSize = chunkSize;

    _visibleLayerDescs = visibleLayerDescs;

    _feedForwardWeights.resize(hiddenWidth * hiddenHeight * visibleLayerDescs.size());
    
    _inputs.resize(visibleLayerDescs.size());

    int hiddenChunksInX = hiddenWidth / chunkSize;
    int hiddenChunksInY = hiddenHeight / chunkSize;

    _hiddenStates.resize(hiddenChunksInX * hiddenChunksInY, 0);

    _predictionWeights.resize(numFeedBack);

    std::uniform_real_distribution<float> initWeightDist(-0.01f, 0.0f);

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        _inputs[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);
        
        int forwardVecSize = _visibleLayerDescs[v]._radius * 2 + 1;

        forwardVecSize *= forwardVecSize;

        for (int x = 0; x < hiddenWidth; x++)
            for (int y = 0; y < hiddenHeight; y++) {
                int hIndex = x + y * hiddenWidth;
                
                int i = v + visibleLayerDescs.size() * hIndex;

                _feedForwardWeights[i].resize(forwardVecSize);

                for (int j = 0; j < forwardVecSize; j++)
                    _feedForwardWeights[i][j] = initWeightDist(rng);
            }

        for (int f = 0; f < numFeedBack; f++) {
            _predictionWeights[f].resize(hiddenWidth * hiddenHeight * visibleLayerDescs.size());

            if (_visibleLayerDescs[v]._predict) {
                for (int x = 0; x < hiddenWidth; x++)
                    for (int y = 0; y < hiddenHeight; y++) {
                        int hIndex = x + y * hiddenWidth;
                        
                        int i = v + visibleLayerDescs.size() * hIndex;

                        _predictionWeights[f][i].resize(forwardVecSize);

                        for (int j = 0; j < forwardVecSize; j++)
                            _predictionWeights[f][i][j] = initWeightDist(rng);
                    }
            }
        }
    }

    _inputsPrev = _inputs;
    _predictions = _inputs;

	_feedBack.resize(numFeedBack);

	for (int f = 0; f < numFeedBack; f++)
		_feedBack[f].resize(hiddenChunksInX * hiddenChunksInY, 0);

    _feedBackPrev = _feedBack;
}

void Layer::createFromStream(std::istream &s) {
    s >> _hiddenWidth >> _hiddenHeight >> _chunkSize;

    int numVisibleLayerDescs;
    s >> numVisibleLayerDescs;

    _visibleLayerDescs.resize(numVisibleLayerDescs);

    for (int v = 0; v < numVisibleLayerDescs; v++) {
        s >> _visibleLayerDescs[v]._width >> _visibleLayerDescs[v]._height >> _visibleLayerDescs[v]._chunkSize;
        s >> _visibleLayerDescs[v]._radius >> _visibleLayerDescs[v]._radius;

        int predict;
        s >> predict;

        _visibleLayerDescs[v]._predict = predict;
    }

    int numFeedForwardWeightSets;
    s >> numFeedForwardWeightSets;

    _feedForwardWeights.resize(numFeedForwardWeightSets);
    
    _inputs.resize(_visibleLayerDescs.size());
    _inputsPrev.resize(_visibleLayerDescs.size());

    _predictions.resize(_visibleLayerDescs.size());

    int hiddenChunksInX = _hiddenWidth / _chunkSize;
    int hiddenChunksInY = _hiddenHeight / _chunkSize;

    _hiddenStates.resize(hiddenChunksInX * hiddenChunksInY);

    // Load
    for (int i = 0; i < _hiddenStates.size(); i++)
        s >> _hiddenStates[i];

    int numFeedBack;
    s >> numFeedBack;

    _predictionWeights.resize(numFeedBack);

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        _inputs[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize));
        _inputsPrev[v].resize(_inputs[v].size());
        _predictions[v].resize(_inputs[v].size());

        // Load
        for (int i = 0; i < _inputs[v].size(); i++)
            s >> _inputs[v][i] >> _inputsPrev[v][i] >> _predictions[v][i];

        int forwardVecSize = _visibleLayerDescs[v]._radius * 2 + 1;

        forwardVecSize *= forwardVecSize;

        for (int x = 0; x < _hiddenWidth; x++)
            for (int y = 0; y < _hiddenHeight; y++) {
                int hIndex = x + y * _hiddenWidth;
                
                int i = v + _visibleLayerDescs.size() * hIndex;

                _feedForwardWeights[i].resize(forwardVecSize);

                for (int j = 0; j < forwardVecSize; j++)
                    s >> _feedForwardWeights[i][j];
            }

        for (int f = 0; f < numFeedBack; f++) {
            _predictionWeights[f].resize(numFeedForwardWeightSets);

            if (_visibleLayerDescs[v]._predict) {
                for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                    for (int y = 0; y < _visibleLayerDescs[v]._height; y++) {
                        int hIndex = x + y * _hiddenWidth;

                        int i = v + _visibleLayerDescs.size() * hIndex;

                        _predictionWeights[f][i].resize(forwardVecSize);

                        for (int j = 0; j < forwardVecSize; j++)
                            s >> _predictionWeights[f][i][j];
                    }
            }
        }
    }

	_feedBack.resize(numFeedBack);
	_feedBackPrev.resize(numFeedBack);

	for (int f = 0; f < numFeedBack; f++) {
		_feedBack[f].resize(hiddenChunksInX * hiddenChunksInY);
		_feedBackPrev[f].resize(_feedBack[f].size());

        // Load
        for (int i = 0; i < _feedBack[f].size(); i++)
            s >> _feedBack[f][i] >> _feedBackPrev[f][i];
    }
}

void Layer::writeToStream(std::ostream &s) {
    s << _hiddenWidth << " " << _hiddenHeight << " " << _chunkSize << std::endl;

    s << _visibleLayerDescs.size() << std::endl;

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        s << _visibleLayerDescs[v]._width << " " << _visibleLayerDescs[v]._height << " " << _visibleLayerDescs[v]._chunkSize << std::endl;
        s << _visibleLayerDescs[v]._radius << " " << _visibleLayerDescs[v]._radius << std::endl;

        s << (_visibleLayerDescs[v]._predict ? 1 : 0) << std::endl;
    }

    s << _feedForwardWeights.size() << std::endl;

    int hiddenChunksInX = _hiddenWidth / _chunkSize;
    int hiddenChunksInY = _hiddenHeight / _chunkSize;

    // Save
    for (int i = 0; i < _hiddenStates.size(); i++)
        s << _hiddenStates[i] << " ";

    s << std::endl;

    s << _feedBack.size() << std::endl;

    int numFeedBack = _feedBack.size();

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        // Save
        for (int i = 0; i < _predictions[v].size(); i++)
            s << _inputs[v][i] << " " << _inputsPrev[v][i] << " " << _predictions[v][i] << " ";

        s << std::endl;

        int forwardVecSize = _visibleLayerDescs[v]._radius * 2 + 1;

        forwardVecSize *= forwardVecSize;

        for (int x = 0; x < _hiddenWidth; x++)
            for (int y = 0; y < _hiddenHeight; y++) {
                int hIndex = x + y * _hiddenWidth;
                
                int i = v + _visibleLayerDescs.size() * hIndex;

                for (int j = 0; j < forwardVecSize; j++)
                    s << _feedForwardWeights[i][j] << " ";

                s << std::endl;
            }

        if (_visibleLayerDescs[v]._predict) {
            for (int f = 0; f < numFeedBack; f++) {
                for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                    for (int y = 0; y < _visibleLayerDescs[v]._height; y++) {
                        int hIndex = x + y * _hiddenWidth;

                        int i = v + _visibleLayerDescs.size() * hIndex;

                        for (int j = 0; j < forwardVecSize; j++)
                            s << _predictionWeights[f][i][j] << " ";

                        s << std::endl;
                    }
            }
        }
    }

	for (int f = 0; f < numFeedBack; f++) {
        // Load
        for (int i = 0; i < _feedBack[f].size(); i++)
            s << _feedBack[f][i] << " " << _feedBackPrev[f][i] << " ";

        s << std::endl;
    }
}

void Layer::forward(const std::vector<std::vector<int>> &inputs, ComputeSystem &cs, float alpha, float gamma) {
    _inputsPrev = _inputs;
    _inputs = inputs;

	_alpha = alpha;
    _gamma = gamma;

    _reconActivationsPrev = _reconActivations;
    _reconCountsPrev = _reconCounts;

    // Clear recon buffers
    _reconActivations.clear();
    _reconActivations.resize(_visibleLayerDescs.size());

    for (int v = 0; v < _visibleLayerDescs.size(); v++)
        _reconActivations[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height, 0.0f);

    _reconCounts = _reconActivations;

    if (_reconActivationsPrev.empty()) {
        _reconActivationsPrev = _reconActivations;
        _reconCountsPrev = _reconCounts;
    }

    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    int numChunks = chunksInX * chunksInY;

    std::uniform_int_distribution<int> seedDist(0, 99999);

    // Queue tasks
    for (int i = 0; i < numChunks; i++) {
        std::shared_ptr<ForwardWorkItem> item = std::make_shared<ForwardWorkItem>();

        item->_hiddenChunkIndex = i;
        item->_pLayer = this;
        item->_rng.seed(seedDist(cs._rng));
			
        cs._pool.addItem(item);
    }

    cs._pool.wait();
}

void Layer::backward(const std::vector<std::vector<int>> &feedBack, ComputeSystem &cs, float beta) {
    _feedBackPrev = _feedBack;
	_feedBack = feedBack;

    _beta = beta;

    _predictionActivationsPrev = _predictionActivations;

    // Clear recon buffers
    _predictionActivations.clear();
    _predictionActivations.resize(_visibleLayerDescs.size());

    for (int v = 0; v < _visibleLayerDescs.size(); v++)
        _predictionActivations[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height, 0.0f);

    _predictionCounts = _predictionActivations;

    if (_predictionActivationsPrev.empty())
        _predictionActivationsPrev = _predictionActivations;

    std::uniform_int_distribution<int> seedDist(0, 99999);

    {
        int chunksInX = _hiddenWidth / _chunkSize;
        int chunksInY = _hiddenHeight / _chunkSize;

        int numChunks = chunksInX * chunksInY;

        std::uniform_int_distribution<int> seedDist(0, 99999);

        // Queue tasks
        for (int i = 0; i < numChunks; i++) {
            std::shared_ptr<BackwardWorkItem> item = std::make_shared<BackwardWorkItem>();

            item->_hiddenChunkIndex = i;
            item->_pLayer = this;
            item->_rng.seed(seedDist(cs._rng));
                
            cs._pool.addItem(item);
        }

        cs._pool.wait();
    }

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        if (!_visibleLayerDescs[v]._predict)
            continue;

        int chunksInX = _visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize;
        int chunksInY = _visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize;

        int numChunks = chunksInX * chunksInY;

        // Queue tasks
        for (int i = 0; i < numChunks; i++) {
            std::shared_ptr<PredictionWorkItem> item = std::make_shared<PredictionWorkItem>();

            item->_visibleChunkIndex = i;
            item->_visibleLayerIndex = v;
            item->_pLayer = this;
            item->_rng.seed(seedDist(cs._rng));
			
            cs._pool.addItem(item);
        }
    }

    cs._pool.wait();
}