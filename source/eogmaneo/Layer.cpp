// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
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

        int spatialVisibleRadius = _pLayer->_visibleLayerDescs[v]._forwardRadius;

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

                                float recon = _pLayer->_reconActivationsPrev[v][vIndex].first / std::max(1.0f, _pLayer->_reconActivationsPrev[v][vIndex].second);

                                _pLayer->_feedForwardWeights[iPrev][wi] += _pLayer->_alpha * (target - std::tanh(recon));
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

        int spatialVisibleRadius = _pLayer->_visibleLayerDescs[v]._forwardRadius;

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
                                _pLayer->_reconActivations[v][vIndex].first += _pLayer->_feedForwardWeights[i][wi];
                                _pLayer->_reconActivations[v][vIndex].second += 1.0f;
                            }
                        }
                }
            }
    }
}

void BackwardWorkItem::run(size_t threadIndex) {
    assert(_pLayer != nullptr);

    int v = _visibleLayerIndex;

    int visibleWidth = _pLayer->_visibleLayerDescs[v]._width;
    int visibleHeight = _pLayer->_visibleLayerDescs[v]._height;

    int visibleChunkSize = _pLayer->_visibleLayerDescs[v]._chunkSize;

    int hiddenChunkSize = _pLayer->_chunkSize;

    int visibleChunksInX = visibleWidth / visibleChunkSize;
    int visibleChunksInY = visibleHeight / visibleChunkSize;

    int visibleChunkX = _visibleChunkIndex % visibleChunksInX;
    int visibleChunkY = _visibleChunkIndex / visibleChunksInX;

    int hiddenChunksInX = _pLayer->_hiddenWidth / hiddenChunkSize;
    int hiddenChunksInY = _pLayer->_hiddenHeight / hiddenChunkSize;

    // Extract input views
    std::vector<float> chunkActivations(visibleChunkSize * visibleChunkSize, 0.0f);
    std::vector<float> chunkActivationsPrev(chunkActivations.size(), 0.0f);
    float chunkDiv = 0.0f;
    float chunkDivPrev = 0.0f;

    int spatialHiddenRadius = _pLayer->_visibleLayerDescs[v]._backwardRadius;

    int spatialHiddenDiam = spatialHiddenRadius * 2 + 1;

    int spatialChunkRadius = std::ceil(static_cast<float>(spatialHiddenRadius) / static_cast<float>(hiddenChunkSize));

    float toInputX = static_cast<float>(hiddenChunksInX) / static_cast<float>(visibleChunksInX);
    float toInputY = static_cast<float>(hiddenChunksInY) / static_cast<float>(visibleChunksInY);

    int hiddenChunkCenterX = (visibleChunkX + 0.5f) * toInputX;
    int hiddenChunkCenterY = (visibleChunkY + 0.5f) * toInputY;

    int hiddenCenterX = (hiddenChunkCenterX + 0.5f) * hiddenChunkSize;
    int hiddenCenterY = (hiddenChunkCenterY + 0.5f) * hiddenChunkSize;

    int lowerHiddenX = hiddenCenterX - spatialHiddenRadius;
    int lowerHiddenY = hiddenCenterY - spatialHiddenRadius;

    int upperHiddenX = hiddenCenterX + spatialHiddenRadius;
    int upperHiddenY = hiddenCenterY + spatialHiddenRadius;

    for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
        for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
            int cx = hiddenChunkCenterX + dcx;
            int cy = hiddenChunkCenterY + dcy;

            if (cx >= 0 && cx < hiddenChunksInX && cy >= 0 && cy < hiddenChunksInY) {
                int hiddenChunkIndex = cx + cy * hiddenChunksInX;

                int maxIndex = _pLayer->_hiddenStates[hiddenChunkIndex];
                int maxIndexPrev = _pLayer->_hiddenStatesPrev[hiddenChunkIndex];
                
                int mdx = maxIndex % hiddenChunkSize;
                int mdy = maxIndex / hiddenChunkSize;

                int hx = cx * hiddenChunkSize + mdx;
                int hy = cy * hiddenChunkSize + mdy;

                int mdxPrev = maxIndexPrev % hiddenChunkSize;
                int mdyPrev = maxIndexPrev / hiddenChunkSize;

                int hxPrev = cx * hiddenChunkSize + mdxPrev;
                int hyPrev = cy * hiddenChunkSize + mdyPrev;

                if (!_pLayer->_feedBack.empty()) {
                    int feedBackIndex = _pLayer->_feedBack[hiddenChunkIndex];
                    int feedBackIndexPrev = _pLayer->_feedBackPrev[hiddenChunkIndex];

                    int fdx = feedBackIndex % hiddenChunkSize;
                    int fdy = feedBackIndex / hiddenChunkSize;

                    int fx = cx * hiddenChunkSize + fdx;
                    int fy = cy * hiddenChunkSize + fdy;

                    int fdxPrev = feedBackIndexPrev % hiddenChunkSize;
                    int fdyPrev = feedBackIndexPrev / hiddenChunkSize;

                    int fxPrev = cx * hiddenChunkSize + fdxPrev;
                    int fyPrev = cy * hiddenChunkSize + fdyPrev;

                    if (fx >= lowerHiddenX && fx <= upperHiddenX && fy >= lowerHiddenY && fy <= upperHiddenY) {
                        int wi = (fx - lowerHiddenX) + (fy - lowerHiddenY) * spatialHiddenDiam;

                        for (int c = 0; c < chunkActivations.size(); c++) {
                            int dvx = c % visibleChunkSize;
                            int dvy = c / visibleChunkSize;

                            int ivIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;
        
                            chunkActivations[c] += _pLayer->_feedBackWeights[v][ivIndex][wi].first;
                        }

                        chunkDiv += 1.0f;
                    }

                    if (fxPrev >= lowerHiddenX && fxPrev <= upperHiddenX && fyPrev >= lowerHiddenY && fyPrev <= upperHiddenY) {
                        int wi = (fxPrev - lowerHiddenX) + (fyPrev - lowerHiddenY) * spatialHiddenDiam;

                        for (int c = 0; c < chunkActivations.size(); c++) {
                            int dvx = c % visibleChunkSize;
                            int dvy = c / visibleChunkSize;

                            int ivIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;
        
                            chunkActivationsPrev[c] += _pLayer->_feedBackWeights[v][ivIndex][wi].first;
                        }

                        chunkDivPrev += 1.0f;
                    }
                }

                if (hx >= lowerHiddenX && hx <= upperHiddenX && hy >= lowerHiddenY && hy <= upperHiddenY) {
                    int wi = (hx - lowerHiddenX) + (hy - lowerHiddenY) * spatialHiddenDiam;
                    
                    for (int c = 0; c < chunkActivations.size(); c++) {
                        int dvx = c % visibleChunkSize;
                        int dvy = c / visibleChunkSize;

                        int ivIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

                        chunkActivations[c] += _pLayer->_feedBackWeights[v][ivIndex][wi].second;
                    }

                    chunkDiv += 1.0f;
                }

                if (hxPrev >= lowerHiddenX && hxPrev <= upperHiddenX && hyPrev >= lowerHiddenY && hyPrev <= upperHiddenY) {
                    int wi = (hxPrev - lowerHiddenX) + (hyPrev - lowerHiddenY) * spatialHiddenDiam;
                    
                    for (int c = 0; c < chunkActivations.size(); c++) {
                        int dvx = c % visibleChunkSize;
                        int dvy = c / visibleChunkSize;

                        int ivIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

                        chunkActivationsPrev[c] += _pLayer->_feedBackWeights[v][ivIndex][wi].second;
                    }

                    chunkDivPrev += 1.0f;
                }
            }
        }

    int targetIndex = _pLayer->_inputs[v][_visibleChunkIndex];

    int dvxTarget = targetIndex % visibleChunkSize;
    int dvyTarget = targetIndex / visibleChunkSize;

    int ivIndexTarget = (visibleChunkX * visibleChunkSize + dvxTarget) + (visibleChunkY * visibleChunkSize + dvyTarget) * visibleWidth;

    int predIndex = 0;

    for (int c = 0; c < chunkActivations.size(); c++) {
        chunkActivations[c] = sigmoid(chunkActivations[c] / std::max(1.0f, chunkDiv));
        chunkActivationsPrev[c] = sigmoid(chunkActivationsPrev[c] / std::max(1.0f, chunkDivPrev));
        
        if (chunkActivations[c] > chunkActivations[predIndex])
            predIndex = c;
    }

    _pLayer->_predictions[v][_visibleChunkIndex] = predIndex;

    for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
        for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
            int cx = hiddenChunkCenterX + dcx;
            int cy = hiddenChunkCenterY + dcy;

            if (cx >= 0 && cx < hiddenChunksInX && cy >= 0 && cy < hiddenChunksInY) {
                int hiddenChunkIndex = cx + cy * hiddenChunksInX;

                int maxIndexPrev = _pLayer->_hiddenStatesPrev[hiddenChunkIndex];

                int mdxPrev = maxIndexPrev % hiddenChunkSize;
                int mdyPrev = maxIndexPrev / hiddenChunkSize;

                int hxPrev = cx * hiddenChunkSize + mdxPrev;
                int hyPrev = cy * hiddenChunkSize + mdyPrev;

                if (!_pLayer->_feedBack.empty()) {
                    int feedBackIndexPrev = _pLayer->_feedBackPrev[hiddenChunkIndex];
                
                    int fdxPrev = feedBackIndexPrev % hiddenChunkSize;
                    int fdyPrev = feedBackIndexPrev / hiddenChunkSize;

                    int fxPrev = cx * hiddenChunkSize + fdxPrev;
                    int fyPrev = cy * hiddenChunkSize + fdyPrev;

                    if (fxPrev >= lowerHiddenX && fxPrev <= upperHiddenX && fyPrev >= lowerHiddenY && fyPrev <= upperHiddenY) {
                        int wi = (fxPrev - lowerHiddenX) + (fyPrev - lowerHiddenY) * spatialHiddenDiam;

                        for (int c = 0; c < chunkActivations.size(); c++) {
                            int dvx = c % visibleChunkSize;
                            int dvy = c / visibleChunkSize;

                            int ivIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;
        
                            float target = (c == targetIndex ? 1.0f : 0.0f);

                            _pLayer->_feedBackWeights[v][ivIndex][wi].first += _pLayer->_beta * (target - chunkActivationsPrev[c]);
                        }
                    }
                }
                
                if (hxPrev >= lowerHiddenX && hxPrev <= upperHiddenX && hyPrev >= lowerHiddenY && hyPrev <= upperHiddenY) {
                    int wi = (hxPrev - lowerHiddenX) + (hyPrev - lowerHiddenY) * spatialHiddenDiam;
                    
                    for (int c = 0; c < chunkActivations.size(); c++) {
                        int dvx = c % visibleChunkSize;
                        int dvy = c / visibleChunkSize;

                        int ivIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

                        float target = (c == targetIndex ? 1.0f : 0.0f);
                        
                        _pLayer->_feedBackWeights[v][ivIndex][wi].second += _pLayer->_beta * (target - chunkActivationsPrev[c]);
                    }
                }
            }
        }
}

void Layer::create(int hiddenWidth, int hiddenHeight, int chunkSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs, unsigned long seed) {
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

    std::uniform_real_distribution<float> initWeightDist(0.0f, 0.01f);

    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        _inputs[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);

        int forwardVecSize = _visibleLayerDescs[v]._forwardRadius * 2 + 1;

        forwardVecSize *= forwardVecSize;

        for (int x = 0; x < hiddenWidth; x++)
            for (int y = 0; y < hiddenHeight; y++) {
                int hIndex = x + y * hiddenWidth;
                
                int i = v + visibleLayerDescs.size() * hIndex;

                _feedForwardWeights[i].resize(forwardVecSize);

                for (int j = 0; j < forwardVecSize; j++)
                    _feedForwardWeights[i][j] = 1.0f - initWeightDist(rng);
            }
    }

    _feedBackWeights.resize(_visibleLayerDescs.size());

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        if (_visibleLayerDescs[v]._predict) {
            _feedBackWeights[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height);

            int backwardVecSize = _visibleLayerDescs[v]._backwardRadius * 2 + 1;

            backwardVecSize *= backwardVecSize;

            for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                for (int y = 0; y < _visibleLayerDescs[v]._height; y++) {
                    int ivIndex = x + y * _visibleLayerDescs[v]._width;
                    
                    _feedBackWeights[v][ivIndex].resize(backwardVecSize);

                    for (int j = 0; j < backwardVecSize; j++)
                        _feedBackWeights[v][ivIndex][j] = std::pair<float, float>(initWeightDist(rng), initWeightDist(rng));
                }
        }
    }

    _hiddenStatesPrev = _hiddenStates;

    _feedBackPrev = _feedBack = _hiddenStates;

    _inputsPrev = _inputs;
    _predictions = _inputs;
}

void Layer::createFromStream(std::istream &s) {
    s >> _hiddenWidth >> _hiddenHeight >> _chunkSize;

    int numVisibleLayerDescs;
    s >> numVisibleLayerDescs;

    _visibleLayerDescs.resize(numVisibleLayerDescs);

    for (int v = 0; v < numVisibleLayerDescs; v++) {
        s >> _visibleLayerDescs[v]._width >> _visibleLayerDescs[v]._height >> _visibleLayerDescs[v]._chunkSize;
        s >> _visibleLayerDescs[v]._forwardRadius >> _visibleLayerDescs[v]._backwardRadius;

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
    _hiddenStatesPrev.resize(_hiddenStates.size());

    // Load
    for (int i = 0; i < _hiddenStates.size(); i++)
        s >> _hiddenStates[i] >> _hiddenStatesPrev[i];
        
    _feedBack.resize(_hiddenStates.size());
    _feedBackPrev.resize(_hiddenStates.size());

    for (int i = 0; i < _feedBack.size(); i++)
        s >> _feedBack[i] >> _feedBackPrev[i];

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        _inputs[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize));
        _inputsPrev[v].resize(_inputs[v].size());
        _predictions[v].resize(_inputs[v].size());

        // Load
        for (int i = 0; i < _inputs[v].size(); i++)
            s >> _inputs[v][i] >> _inputsPrev[v][i] >> _predictions[v][i];

        int forwardVecSize = _visibleLayerDescs[v]._forwardRadius * 2 + 1;

        forwardVecSize *= forwardVecSize;

        for (int x = 0; x < _hiddenWidth; x++)
            for (int y = 0; y < _hiddenHeight; y++) {
                int hIndex = x + y * _hiddenWidth;
                
                int i = v + _visibleLayerDescs.size() * hIndex;

                _feedForwardWeights[i].resize(forwardVecSize);

                for (int j = 0; j < forwardVecSize; j++)
                    s >> _feedForwardWeights[i][j];
            }
    }

    _feedBackWeights.resize(_visibleLayerDescs.size());

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        if (_visibleLayerDescs[v]._predict) {
            _feedBackWeights[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height);

            int backwardVecSize = _visibleLayerDescs[v]._backwardRadius * 2 + 1;

            backwardVecSize *= backwardVecSize;

            for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                for (int y = 0; y < _visibleLayerDescs[v]._height; y++) {
                    int ivIndex = x + y * _visibleLayerDescs[v]._width;
                    
                    _feedBackWeights[v][ivIndex].resize(backwardVecSize);

                    for (int j = 0; j < backwardVecSize; j++)
                        s >> _feedBackWeights[v][ivIndex][j].first >> _feedBackWeights[v][ivIndex][j].second;
                }
        }
    }
}

void Layer::writeToStream(std::ostream &s) {
    s << _hiddenWidth << " " << _hiddenHeight << " " << _chunkSize << std::endl;

    s << _visibleLayerDescs.size() << std::endl;

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        s << _visibleLayerDescs[v]._width << " " << _visibleLayerDescs[v]._height << " " << _visibleLayerDescs[v]._chunkSize << std::endl;
        s << _visibleLayerDescs[v]._forwardRadius << " " << _visibleLayerDescs[v]._backwardRadius << std::endl;

        s << (_visibleLayerDescs[v]._predict ? 1 : 0) << std::endl;
    }

    s << _feedForwardWeights.size() << std::endl;

    int hiddenChunksInX = _hiddenWidth / _chunkSize;
    int hiddenChunksInY = _hiddenHeight / _chunkSize;

    // Save
    for (int i = 0; i < _hiddenStates.size(); i++)
        s << _hiddenStates[i] << " " << _hiddenStatesPrev[i] << " ";

    s << std::endl;

    for (int i = 0; i < _feedBack.size(); i++)
        s << _feedBack[i] << " " << _feedBackPrev[i] << " ";

    s << std::endl;

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        // Save
        for (int i = 0; i < _predictions[v].size(); i++)
            s << _inputs[v][i] << " " << _inputsPrev[v][i] << " " << _predictions[v][i] << " ";

        s << std::endl;

        int forwardVecSize = _visibleLayerDescs[v]._forwardRadius * 2 + 1;

        forwardVecSize *= forwardVecSize;

        for (int x = 0; x < _hiddenWidth; x++)
            for (int y = 0; y < _hiddenHeight; y++) {
                int hIndex = x + y * _hiddenWidth;
                
                int i = v + _visibleLayerDescs.size() * hIndex;

                for (int j = 0; j < forwardVecSize; j++)
                    s << _feedForwardWeights[i][j] << " ";

                s << std::endl;
            }
    }

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        if (_visibleLayerDescs[v]._predict) {
            int backwardVecSize = _visibleLayerDescs[v]._backwardRadius * 2 + 1;

            backwardVecSize *= backwardVecSize;

            for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                for (int y = 0; y < _visibleLayerDescs[v]._height; y++) {
                    int ivIndex = x + y * _visibleLayerDescs[v]._width;
                    
                    for (int j = 0; j < backwardVecSize; j++)
                        s << _feedBackWeights[v][ivIndex][j].first << " " << _feedBackWeights[v][ivIndex][j].second << " ";

                    s << std::endl;
                }
        }
    }
}

void Layer::forward(const std::vector<std::vector<int>> &inputs, ComputeSystem &cs, float alpha) {
    _inputsPrev = _inputs;
    _inputs = inputs;

    _hiddenStatesPrev = _hiddenStates;

	_alpha = alpha;

    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    int numChunks = chunksInX * chunksInY;

    _reconActivationsPrev = _reconActivations;

    // Clear recon buffers
    _reconActivations.clear();
    _reconActivations.resize(_visibleLayerDescs.size());

    for (int v = 0; v < _visibleLayerDescs.size(); v++)
        _reconActivations[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height, std::pair<float, float>(0.0f, 0.0f));

    if (_reconActivationsPrev.empty())
        _reconActivationsPrev = _reconActivations;

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

void Layer::backward(const std::vector<int> &feedBack, ComputeSystem &cs, float beta) {
    _feedBackPrev = _feedBack;
	_feedBack = feedBack;

    _beta = beta;

    std::uniform_int_distribution<int> seedDist(0, 99999);

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        if (!_visibleLayerDescs[v]._predict)
            continue;

        int chunksInX = _visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize;
        int chunksInY = _visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize;

        int numChunks = chunksInX * chunksInY;

        // Queue tasks
        for (int i = 0; i < numChunks; i++) {
            std::shared_ptr<BackwardWorkItem> item = std::make_shared<BackwardWorkItem>();

            item->_visibleChunkIndex = i;
            item->_visibleLayerIndex = v;
            item->_pLayer = this;
            item->_rng.seed(seedDist(cs._rng));
			
            cs._pool.addItem(item);
        }
    }

    cs._pool.wait();
}

std::vector<float> Layer::getFeedBackWeights(int v, int f, int x, int y) const {
    std::vector<float> weights;

    int vu = v + f * _visibleLayerDescs.size();

    // Reverse project
    int hIndex = x + y * _hiddenWidth;

    int hiddenChunkSize = _chunkSize;
    
    int visibleBitsPerChunk = _visibleLayerDescs[v]._chunkSize * _visibleLayerDescs[v]._chunkSize;

    int hiddenChunksInX = _hiddenWidth / hiddenChunkSize;
    int hiddenChunksInY = _hiddenHeight / hiddenChunkSize;

    int hiddenChunkX = x / hiddenChunksInX;
    int hiddenChunkY = y / hiddenChunksInY;

    // Project unit
    int visibleChunkSize = _visibleLayerDescs[v]._chunkSize;

    int visibleChunksInX = _visibleLayerDescs[v]._width / visibleChunkSize;
    int visibleChunksInY = _visibleLayerDescs[v]._height / visibleChunkSize;

    float toInputX1 = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
    float toInputY1 = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

    int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX1;
    int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY1;

    int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
    int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

    int spatialReverseHiddenRadius = std::ceil(_visibleLayerDescs[v]._backwardRadius / std::min(toInputX1, toInputY1)) + 2;

    int spatialReverseHiddenDiam = spatialReverseHiddenRadius * 2 + 1;

    weights.resize(spatialReverseHiddenDiam * spatialReverseHiddenDiam, 0.0f);

    int spatialChunkRadius1 = std::ceil(static_cast<float>(spatialReverseHiddenRadius) / static_cast<float>(visibleChunkSize));

    int lowerVisibleX = visibleCenterX - spatialReverseHiddenRadius;
    int lowerVisibleY = visibleCenterY - spatialReverseHiddenRadius;

    int upperVisibleX = visibleCenterX + spatialReverseHiddenRadius;
    int upperVisibleY = visibleCenterY + spatialReverseHiddenRadius;

    for (int dcx = -spatialChunkRadius1; dcx <= spatialChunkRadius1; dcx++)
        for (int dcy = -spatialChunkRadius1; dcy <= spatialChunkRadius1; dcy++) {
            int cx = visibleChunkCenterX + dcx;
            int cy = visibleChunkCenterY + dcy;

            if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                int visibleChunkIndex = cx + cy * visibleChunksInX;

                // Project this chunk back to hidden
                int spatialHiddenRadius = _visibleLayerDescs[v]._backwardRadius;

                int spatialHiddenDiam = spatialHiddenRadius * 2 + 1;

                float toInputX2 = static_cast<float>(hiddenChunksInX) / static_cast<float>(visibleChunksInX);
                float toInputY2 = static_cast<float>(hiddenChunksInY) / static_cast<float>(visibleChunksInY);

                int hiddenChunkCenterX = (cx + 0.5f) * toInputX2;
                int hiddenChunkCenterY = (cy + 0.5f) * toInputY2;

                int hiddenCenterX = (hiddenChunkCenterX + 0.5f) * hiddenChunkSize;
                int hiddenCenterY = (hiddenChunkCenterY + 0.5f) * hiddenChunkSize;

                int lowerHiddenX = hiddenCenterX - spatialHiddenRadius;
                int lowerHiddenY = hiddenCenterY - spatialHiddenRadius;

                int upperHiddenX = hiddenCenterX + spatialHiddenRadius;
                int upperHiddenY = hiddenCenterY + spatialHiddenRadius;

                if (x >= lowerHiddenX && x <= upperHiddenX && y >= lowerHiddenY && y <= upperHiddenY) {
                    for (int c = 0; c < visibleBitsPerChunk; c++) {
                        int mdx = c % visibleChunkSize;
                        int mdy = c / visibleChunkSize;

                        int vx = cx * visibleChunkSize + mdx;
                        int vy = cy * visibleChunkSize + mdy;

                        if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                            int ivIndex = vx + vy * _visibleLayerDescs[v]._width;

                            int wi = (x - lowerHiddenX) + (y - lowerHiddenY) * spatialHiddenDiam;

                            assert(wi >= 0 && wi < _feedBackWeights[v][ivIndex].size());

                            float weight = _feedBackWeights[vu][ivIndex][wi].first;

                            weights[(vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialReverseHiddenDiam] = weight;
                        }
                    }
                }
            }
        }

    return weights;
}