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

void ForwardWorkItem::run(size_t threadIndex) {
    assert(_pLayer != nullptr);

    int hiddenChunkSize = _pLayer->_chunkSize;

    int hiddenChunksInX = _pLayer->_hiddenWidth / hiddenChunkSize;
    int hiddenChunksInY = _pLayer->_hiddenHeight / hiddenChunkSize;

    int hiddenChunkX = _hiddenChunkIndex % hiddenChunksInX;
    int hiddenChunkY = _hiddenChunkIndex / hiddenChunksInX;

    int statePrev = _pLayer->_hiddenStatesPrev[_hiddenChunkIndex];

    // Extract input views
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

        for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
            for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                int cx = visibleChunkCenterX + dcx;
                int cy = visibleChunkCenterY + dcy;

                if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                    int visibleChunkIndex = cx + cy * visibleChunksInX;

                    int maxIndex = _pLayer->_inputs[v][visibleChunkIndex];
                    int maxIndexPrev = _pLayer->_inputsPrev[v][visibleChunkIndex];
 
                    int mdx = maxIndex % visibleChunkSize;
                    int mdy = maxIndex / visibleChunkSize;

                    int mdxPrev = maxIndexPrev % visibleChunkSize;
                    int mdyPrev = maxIndexPrev / visibleChunkSize;

                    int vx = cx * visibleChunkSize + mdx;
                    int vy = cy * visibleChunkSize + mdy;

                    int vxPrev = cx * visibleChunkSize + mdxPrev;
                    int vyPrev = cy * visibleChunkSize + mdyPrev;   

                    if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                        for (int c = 0; c < chunkActivations.size(); c++) {
                            int dhx = c % hiddenChunkSize;
                            int dhy = c / hiddenChunkSize;

                            int i = v + _pLayer->_visibleLayerDescs.size() * ((hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * _pLayer->_hiddenWidth);
                        
                            int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                            chunkActivations[c] += _pLayer->_feedForwardWeights[i][wi];
                        }                 
                    }
                    
                    // Reduce weight on all 0 inputs
                    for (int dvx = 0; dvx < visibleChunkSize; dvx++)
                        for (int dvy = 0; dvy < visibleChunkSize; dvy++) {
                            int index = dvx + dvy * visibleChunkSize;

                            if (index != maxIndexPrev) {
                                int ovx = cx * visibleChunkSize + dvx;
                                int ovy = cy * visibleChunkSize + dvy;
                                
                                if (ovx >= lowerVisibleX && ovx <= upperVisibleX && ovy >= lowerVisibleY && ovy <= upperVisibleY) {
                                    int c = statePrev;
                                    
                                    int dhx = c % hiddenChunkSize;
                                    int dhy = c / hiddenChunkSize;

                                    int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * _pLayer->_hiddenWidth;
                                    
                                    int i = v + _pLayer->_visibleLayerDescs.size() * hIndex;

                                    int wi = (ovx - lowerVisibleX) + (ovy - lowerVisibleY) * spatialVisibleDiam;

                                    float newWeight = std::max(0.0f, _pLayer->_feedForwardWeights[i][wi] - _pLayer->_alpha);

                                    _pLayer->_feedForwardWeights[i][wi] = newWeight;
                                }
                            }
                            // else {
                            //     int ovx = cx * visibleChunkSize + dvx;
                            //     int ovy = cy * visibleChunkSize + dvy;
                                
                            //     if (ovx >= lowerVisibleX && ovx <= upperVisibleX && ovy >= lowerVisibleY && ovy <= upperVisibleY) {
                            //         int c = statePrev;

                            //         int dhx = c % hiddenChunkSize;
                            //         int dhy = c / hiddenChunkSize;

                            //         int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * _pLayer->_hiddenWidth;
                                    
                            //         int i = v + _pLayer->_visibleLayerDescs.size() * hIndex;

                            //         int wi = (ovx - lowerVisibleX) + (ovy - lowerVisibleY) * spatialVisibleDiam;

                            //         float newWeight = std::min(1.0f, _pLayer->_feedForwardWeights[i][wi] + _pLayer->_alpha);

                            //         _pLayer->_feedForwardWeights[i][wi] = newWeight;
                            //     }
                            // }
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
}

void BackwardWorkItem::run(size_t threadIndex) {
    assert(_pLayer != nullptr);

    int v = _visibleLayerIndex;
    
    if (!_pLayer->_visibleLayerDescs[v]._predict)
        return;

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

    // For each feedback layer
    for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
        for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
            int cx = hiddenChunkCenterX + dcx;
            int cy = hiddenChunkCenterY + dcy;

            if (cx >= 0 && cx < hiddenChunksInX && cy >= 0 && cy < hiddenChunksInY) {
                int hiddenChunkIndex = cx + cy * hiddenChunksInX;

                for (int f = 0; f < _pLayer->_feedBack.size(); f++) {
                    int maxIndex = _pLayer->_feedBack[f][hiddenChunkIndex];
                    int maxIndexPrev = _pLayer->_feedBackPrev[f][hiddenChunkIndex];

                    int mdx = maxIndex % hiddenChunkSize;
                    int mdy = maxIndex / hiddenChunkSize;

                    int mdxPrev = maxIndexPrev % hiddenChunkSize;
                    int mdyPrev = maxIndexPrev / hiddenChunkSize;
                    
                    int hx = cx * hiddenChunkSize + mdx;
                    int hy = cy * hiddenChunkSize + mdy;

                    int hxPrev = cx * hiddenChunkSize + mdxPrev;
                    int hyPrev = cy * hiddenChunkSize + mdyPrev;
                    
                    if (hx >= lowerHiddenX && hx <= upperHiddenX && hy >= lowerHiddenY && hy <= upperHiddenY) {
                        for (int c = 0; c < chunkActivations.size(); c++) {
                            int dvx = c % visibleChunkSize;
                            int dvy = c / visibleChunkSize;

                            int vIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

                            int i = f + _pLayer->_feedBack.size() * vIndex;

                            int wi = (hx - lowerHiddenX) + (hy - lowerHiddenY) * spatialHiddenDiam;

                            chunkActivations[c] += _pLayer->_feedBackWeights[v][i][wi];
                        }
                    }

                    if (hxPrev >= lowerHiddenX && hxPrev <= upperHiddenX && hyPrev >= lowerHiddenY && hyPrev <= upperHiddenY) {
                        // Delta
                        {
                            int c = _pLayer->_predictionsPrev[v][_visibleChunkIndex];

                            int dvx = c % visibleChunkSize;
                            int dvy = c / visibleChunkSize;

                            int vIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

                            int i = f + _pLayer->_feedBack.size() * vIndex;

                            int wi = (hxPrev - lowerHiddenX) + (hyPrev - lowerHiddenY) * spatialHiddenDiam;

                            float newWeight = _pLayer->_feedBackWeights[v][i][wi] - _pLayer->_beta;

                            _pLayer->_feedBackWeights[v][i][wi] = newWeight;
                        }

                        // Delta
                        {
                            int c = _pLayer->_inputs[v][_visibleChunkIndex];

                            int dvx = c % visibleChunkSize;
                            int dvy = c / visibleChunkSize;

                            int vIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

                            int i = f + _pLayer->_feedBack.size() * vIndex;

                            int wi = (hxPrev - lowerHiddenX) + (hyPrev - lowerHiddenY) * spatialHiddenDiam;

                            float newWeight = _pLayer->_feedBackWeights[v][i][wi] + _pLayer->_beta;

                            _pLayer->_feedBackWeights[v][i][wi] = newWeight;
                        }
                    }
                }
            }
        }
        
    // Find max element
    int predMaxIndex = 0;
    
    for (int c = 0; c < chunkActivations.size(); c++) {
        if (chunkActivations[c] > chunkActivations[predMaxIndex])
            predMaxIndex = c;
    }
    
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    if (_pLayer->_delta != 0.0f && dist01(_rng) < _pLayer->_epsilon) {
        std::uniform_int_distribution<int> chunkDist(0, chunkActivations.size() - 1);

        _pLayer->_predictions[v][_visibleChunkIndex] = chunkDist(_rng);
    }
    else
        _pLayer->_predictions[v][_visibleChunkIndex] = predMaxIndex;

    float qPrev = chunkActivations[_pLayer->_predictions[v][_visibleChunkIndex]];

    // Update previous replay samples 
    std::vector<float> qs(_pLayer->_replaySamples.size());

    for (int t = 0; t < _pLayer->_replaySamples.size(); t++) {
        ReplaySample &sample = _pLayer->_replaySamples[t];
        
        float q = sample._reward + _pLayer->_gamma * qPrev;

        qs[t] = q;

        qPrev = q;
    }

    // Update previous replay samples 
    if (_pLayer->_delta != 0.0f && !_pLayer->_replaySamples.empty()) {
        std::uniform_int_distribution<int> replayDist(0, _pLayer->_replaySamples.size() - 1);

        for (int iter = 0; iter < _pLayer->_replayIter; iter++) {
            int sampleIndex = replayDist(_rng);
            
            ReplaySample &sample = _pLayer->_replaySamples[sampleIndex];

            int c = sample._predictionsPrev[v][_visibleChunkIndex];
            
            float q = qs[sampleIndex];

            // Activate
            float replayChunkActivation = 0.0f;

            for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
                for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                    int cx = hiddenChunkCenterX + dcx;
                    int cy = hiddenChunkCenterY + dcy;

                    if (cx >= 0 && cx < hiddenChunksInX && cy >= 0 && cy < hiddenChunksInY) {
                        int hiddenChunkIndex = cx + cy * hiddenChunksInX;

                        for (int f = 0; f < _pLayer->_feedBack.size(); f++) {
                            int maxIndexPrev = sample._feedBackPrev[f][hiddenChunkIndex];

                            int mdxPrev = maxIndexPrev % hiddenChunkSize;
                            int mdyPrev = maxIndexPrev / hiddenChunkSize;

                            int hxPrev = cx * hiddenChunkSize + mdxPrev;
                            int hyPrev = cy * hiddenChunkSize + mdyPrev;
                            
                            if (hxPrev >= lowerHiddenX && hxPrev <= upperHiddenX && hyPrev >= lowerHiddenY && hyPrev <= upperHiddenY) {
                                int dvx = c % visibleChunkSize;
                                int dvy = c / visibleChunkSize;

                                int vIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

                                int i = f + _pLayer->_feedBack.size() * vIndex;

                                int wi = (hxPrev - lowerHiddenX) + (hyPrev - lowerHiddenY) * spatialHiddenDiam;

                                replayChunkActivation += _pLayer->_feedBackWeights[v][i][wi];
                            }
                        }
                    }
                }

            // Learn
            for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
                for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                    int cx = hiddenChunkCenterX + dcx;
                    int cy = hiddenChunkCenterY + dcy;

                    if (cx >= 0 && cx < hiddenChunksInX && cy >= 0 && cy < hiddenChunksInY) {
                        int hiddenChunkIndex = cx + cy * hiddenChunksInX;

                        for (int f = 0; f < _pLayer->_feedBack.size(); f++) {
                            int maxIndexPrev = sample._feedBackPrev[f][hiddenChunkIndex];

                            int mdxPrev = maxIndexPrev % hiddenChunkSize;
                            int mdyPrev = maxIndexPrev / hiddenChunkSize;

                            int hxPrev = cx * hiddenChunkSize + mdxPrev;
                            int hyPrev = cy * hiddenChunkSize + mdyPrev;

                            if (hxPrev >= lowerHiddenX && hxPrev <= upperHiddenX && hyPrev >= lowerHiddenY && hyPrev <= upperHiddenY) {
                                // Q learning factor
                                int dvx = c % visibleChunkSize;
                                int dvy = c / visibleChunkSize;

                                int vIndex = (visibleChunkX * visibleChunkSize + dvx) + (visibleChunkY * visibleChunkSize + dvy) * visibleWidth;

                                int i = f + _pLayer->_feedBack.size() * vIndex;

                                int wi = (hxPrev - lowerHiddenX) + (hyPrev - lowerHiddenY) * spatialHiddenDiam;

                                float newWeight = _pLayer->_feedBackWeights[v][i][wi] + _pLayer->_delta * (q - replayChunkActivation);

                                _pLayer->_feedBackWeights[v][i][wi] = newWeight;
                            }
                        }
                    }
                }
        }
    }
}

void Layer::create(int hiddenWidth, int hiddenHeight, int chunkSize, bool hasFeedBack, const std::vector<VisibleLayerDesc> &visibleLayerDescs, unsigned long seed) {
    std::mt19937 rng(seed);

    _hiddenWidth = hiddenWidth;
    _hiddenHeight = hiddenHeight;
    _chunkSize = chunkSize;

    _visibleLayerDescs = visibleLayerDescs;

    _feedForwardWeights.resize(hiddenWidth * hiddenHeight * visibleLayerDescs.size());
    _feedBackWeights.resize(visibleLayerDescs.size());

    _predictions.resize(visibleLayerDescs.size());
    _predictionsPrev.resize(visibleLayerDescs.size());

    _inputs.resize(visibleLayerDescs.size());
    _inputsPrev.resize(visibleLayerDescs.size());

    int hiddenChunksInX = hiddenWidth / chunkSize;
    int hiddenChunksInY = hiddenHeight / chunkSize;

    _hiddenStates.resize(hiddenChunksInX * hiddenChunksInY, 0);
    _hiddenStatesPrev.resize(hiddenChunksInX * hiddenChunksInY, 0);
    
    int numFeedBack = hasFeedBack ? 2 : 1;

    std::uniform_real_distribution<float> initWeightDistHigh(0.99f, 1.0f);
    std::uniform_real_distribution<float> initWeightDistLow(-0.01f, 0.01f);
    
    for (int v = 0; v < visibleLayerDescs.size(); v++) {
        _predictions[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);
        _predictionsPrev[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);

        _inputs[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);
        _inputsPrev[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);

        int forwardVecSize = _visibleLayerDescs[v]._forwardRadius * 2 + 1;

        forwardVecSize *= forwardVecSize;

        for (int x = 0; x < hiddenWidth; x++)
            for (int y = 0; y < hiddenHeight; y++) {
                int hIndex = x + y * hiddenWidth;
                
                int i = v + visibleLayerDescs.size() * hIndex;

                _feedForwardWeights[i].resize(forwardVecSize);

                for (int j = 0; j < forwardVecSize; j++)
                    _feedForwardWeights[i][j] = initWeightDistHigh(rng);
            }

        int backwardVecSize = _visibleLayerDescs[v]._backwardRadius * 2 + 1;

        backwardVecSize *= backwardVecSize;

        if (_visibleLayerDescs[v]._predict) {
            _feedBackWeights[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height * numFeedBack);

            for (int f = 0; f < numFeedBack; f++) {
                for (int x = 0; x < visibleLayerDescs[v]._width; x++)
                    for (int y = 0; y < visibleLayerDescs[v]._height; y++) {
                        int vIndex = x + y * visibleLayerDescs[v]._width;
                        
                        int i = f + numFeedBack * vIndex;

                        _feedBackWeights[v][i].resize(backwardVecSize);

                        for (int j = 0; j < backwardVecSize; j++)
                            _feedBackWeights[v][i][j] = initWeightDistLow(rng);
                    }
            }
        }
    }

	_feedBack.resize(numFeedBack);
	_feedBackPrev.resize(numFeedBack);

	for (int f = 0; f < numFeedBack; f++) {
		_feedBack[f].resize(hiddenChunksInX * hiddenChunksInY, 0);
		_feedBackPrev[f].resize(hiddenChunksInX * hiddenChunksInY, 0);
    }
}

void Layer::createFromStream(std::istream &s) {
    s >> _hiddenWidth >> _hiddenHeight >> _chunkSize;

    int numVisibleLayerDescs;
    s >> numVisibleLayerDescs;

    _visibleLayerDescs.resize(numVisibleLayerDescs);

    for (int v = 0; v < numVisibleLayerDescs; v++) {
        s >> _visibleLayerDescs[v]._width >> _visibleLayerDescs[v]._height >> _visibleLayerDescs[v]._chunkSize;
        s >> _visibleLayerDescs[v]._forwardRadius >> _visibleLayerDescs[v]._backwardRadius;
        s >> _visibleLayerDescs[v]._predict;
    }

    int numFeedForwardWeightSets;
    s >> numFeedForwardWeightSets;

    _feedForwardWeights.resize(numFeedForwardWeightSets);
    _feedBackWeights.resize(_visibleLayerDescs.size());

    _predictions.resize(_visibleLayerDescs.size());
    _predictionsPrev.resize(_visibleLayerDescs.size());

    _inputs.resize(_visibleLayerDescs.size());
    _inputsPrev.resize(_visibleLayerDescs.size());

    int hiddenChunksInX = _hiddenWidth / _chunkSize;
    int hiddenChunksInY = _hiddenHeight / _chunkSize;

    _hiddenStates.resize(hiddenChunksInX * hiddenChunksInY, 0);
    _hiddenStatesPrev.resize(hiddenChunksInX * hiddenChunksInY, 0);
    
    // Load
    for (int i = 0; i < _hiddenStates.size(); i++)
        s >> _hiddenStates[i] >> _hiddenStatesPrev[i];

    bool hasFeedBack;
    s >> hasFeedBack;

    int numFeedBack = hasFeedBack ? 2 : 1;

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        _predictions[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);
        _predictionsPrev[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);

        _inputs[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);
        _inputsPrev[v].resize((_visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize) * (_visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize), 0);

        // Load 
        for (int i = 0; i < _predictions[v].size(); i++)
            s >> _predictions[v][i] >> _predictionsPrev[v][i];

        for (int i = 0; i < _inputs[v].size(); i++)
            s >> _inputs[v][i] >> _inputsPrev[v][i];

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

        int backwardVecSize = _visibleLayerDescs[v]._backwardRadius * 2 + 1;

        backwardVecSize *= backwardVecSize;

        if (_visibleLayerDescs[v]._predict) {
            _feedBackWeights[v].resize(_visibleLayerDescs[v]._width * _visibleLayerDescs[v]._height * numFeedBack);

            for (int f = 0; f < numFeedBack; f++) {
                for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                    for (int y = 0; y < _visibleLayerDescs[v]._height; y++) {
                        int vIndex = x + y * _visibleLayerDescs[v]._width;
                        
                        int i = f + numFeedBack * vIndex;

                        _feedBackWeights[v][i].resize(backwardVecSize);

                        for (int j = 0; j < backwardVecSize; j++)
                            s >> _feedBackWeights[v][i][j];
                    }
            }
        }
    }

	_feedBack.resize(numFeedBack);
	_feedBackPrev.resize(numFeedBack);

	for (int f = 0; f < numFeedBack; f++) {
		_feedBack[f].resize(hiddenChunksInX * hiddenChunksInY, 0);
		_feedBackPrev[f].resize(hiddenChunksInX * hiddenChunksInY, 0);

        // Load
        for (int i = 0; i < _feedBack[f].size(); i++)
            s >> _feedBack[f][i] >> _feedBackPrev[f][i];
    }

    s >> _maxReplaySamples;

    int numSamples;
    s >> numSamples;

    _replaySamples.resize(numSamples);

    for (int t = 0; t < _replaySamples.size(); t++) {
        s >> _replaySamples[t]._reward;

        _replaySamples[t]._predictionsPrev.resize(_predictionsPrev.size());

        for (int v = 0; v < _predictionsPrev.size(); v++) {
            _replaySamples[t]._predictionsPrev[v].resize(_predictionsPrev[v].size());

            for (int i = 0; i < _predictionsPrev[v].size(); i++)
                s >> _replaySamples[t]._predictionsPrev[v][i];
        }

        _replaySamples[t]._feedBackPrev.resize(_feedBackPrev.size());

        for (int f = 0; f < _feedBackPrev.size(); f++) {
            _replaySamples[t]._feedBackPrev[f].resize(_feedBackPrev[f].size());

            for (int i = 0; i < _feedBackPrev[f].size(); i++)
                s >> _replaySamples[t]._feedBackPrev[f][i];
        }
    }
}

void Layer::writeToStream(std::ostream &s) {
    s << _hiddenWidth << " " << _hiddenHeight << " " << _chunkSize << std::endl;

    s << _visibleLayerDescs.size() << std::endl;

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        s << _visibleLayerDescs[v]._width << " " << _visibleLayerDescs[v]._height << " " << _visibleLayerDescs[v]._chunkSize << std::endl;
        s << _visibleLayerDescs[v]._forwardRadius << " " << _visibleLayerDescs[v]._backwardRadius << std::endl;
        s << _visibleLayerDescs[v]._predict << std::endl;
    }

    s << _feedForwardWeights.size() << std::endl;

    int hiddenChunksInX = _hiddenWidth / _chunkSize;
    int hiddenChunksInY = _hiddenHeight / _chunkSize;

    // Save
    for (int i = 0; i < _hiddenStates.size(); i++)
        s << _hiddenStates[i] << " " << _hiddenStatesPrev[i] << std::endl;

    s << (_feedBack.size() > 1) << std::endl;

    int numFeedBack = _feedBack.size();

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        // Save 
        for (int i = 0; i < _predictions[v].size(); i++)
            s << _predictions[v][i] << " " << _predictionsPrev[v][i] << std::endl;

        for (int i = 0; i < _inputs[v].size(); i++)
            s << _inputs[v][i] << " " << _inputsPrev[v][i] << std::endl;

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

        int backwardVecSize = _visibleLayerDescs[v]._backwardRadius * 2 + 1;

        backwardVecSize *= backwardVecSize;

        if (_visibleLayerDescs[v]._predict) {
            for (int f = 0; f < numFeedBack; f++) {
                for (int x = 0; x < _visibleLayerDescs[v]._width; x++)
                    for (int y = 0; y < _visibleLayerDescs[v]._height; y++) {
                        int vIndex = x + y * _visibleLayerDescs[v]._width;
                        
                        int i = f + numFeedBack * vIndex;

                        for (int j = 0; j < backwardVecSize; j++)
                            s << _feedBackWeights[v][i][j] << " ";

                        s << std::endl;
                    }
            }
        }
    }

	for (int f = 0; f < numFeedBack; f++) {
        // Load
        for (int i = 0; i < _feedBack[f].size(); i++)
            s << _feedBack[f][i] << " " << _feedBackPrev[f][i] << std::endl;
    }

    s << _maxReplaySamples << std::endl;

    s << _replaySamples.size() << std::endl;

    for (int t = 0; t < _replaySamples.size(); t++) {
        s << _replaySamples[t]._reward << std::endl;

        for (int v = 0; v < _replaySamples[t]._predictionsPrev.size(); v++) {
            for (int i = 0; i < _replaySamples[t]._predictionsPrev[v].size(); i++)
                s << _replaySamples[t]._predictionsPrev[v][i] << " ";
        }

        s << std::endl;

        for (int f = 0; f < _replaySamples[t]._feedBackPrev.size(); f++) {
            for (int i = 0; i < _replaySamples[t]._feedBackPrev[f].size(); i++)
                s << _replaySamples[t]._feedBackPrev[f][i] << " ";
        }

        s << std::endl;
    }
}

void Layer::forward(const std::vector<std::vector<int>> &inputs, ComputeSystem &system, float alpha) {
    _inputsPrev = _inputs;
    _inputs = inputs;

    _hiddenStatesPrev = _hiddenStates;

	_alpha = alpha;

    int chunksInX = _hiddenWidth / _chunkSize;
    int chunksInY = _hiddenHeight / _chunkSize;

    int numChunks = chunksInX * chunksInY;

    std::uniform_int_distribution<int> seedDist(0, 99999);

    // Queue tasks
    for (int i = 0; i < numChunks; i++) {
        std::shared_ptr<ForwardWorkItem> item = std::make_shared<ForwardWorkItem>();

        item->_hiddenChunkIndex = i;
        item->_pLayer = this;
        item->_rng.seed(seedDist(system._rng));
			
        system._pool.addItem(item);
    }

    system._pool.wait();
}

void Layer::backward(const std::vector<std::vector<int>> &feedBack, ComputeSystem &system, float reward, float beta, float delta, float gamma, float epsilon) {
    _feedBackPrev = _feedBack;
	_feedBack = feedBack;

    _predictionsPrev = _predictions;
    
    // Add new replay sample
    ReplaySample sample;

    sample._predictionsPrev = _predictionsPrev;
    sample._feedBackPrev = _feedBackPrev;
    sample._reward = reward;

    _replaySamples.insert(_replaySamples.begin(), sample);
    
    if (_replaySamples.size() > _maxReplaySamples)
        _replaySamples.resize(_maxReplaySamples);

    _reward = reward;
    _beta = beta;
    _delta = delta;
    _gamma = gamma;
    _epsilon = epsilon;
    
    std::uniform_int_distribution<int> seedDist(0, 99999);

    for (int v = 0; v < _visibleLayerDescs.size(); v++) {
        int chunksInX = _visibleLayerDescs[v]._width / _visibleLayerDescs[v]._chunkSize;
        int chunksInY = _visibleLayerDescs[v]._height / _visibleLayerDescs[v]._chunkSize;

        int numChunks = chunksInX * chunksInY;

        // Queue tasks
        for (int i = 0; i < numChunks; i++) {
            std::shared_ptr<BackwardWorkItem> item = std::make_shared<BackwardWorkItem>();

            item->_visibleChunkIndex = i;
            item->_visibleLayerIndex = v;
            item->_pLayer = this;
            item->_rng.seed(seedDist(system._rng));
			
            system._pool.addItem(item);
        }
    }

    system._pool.wait();
}

std::vector<float> Layer::getFeedBackWeights(int v, int f, int x, int y) const {
    std::vector<float> weights;

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
                            int vIndex = vx + vy * _visibleLayerDescs[v]._width;

                            int i = f + _feedBack.size() * vIndex;

                            int wi = (x - lowerHiddenX) + (y - lowerHiddenY) * spatialHiddenDiam;

                            assert(wi >= 0 && wi < _feedBackWeights[v][i].size());

                            float weight = _feedBackWeights[v][i][wi];

                            weights[(vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialReverseHiddenDiam] = weight;
                        }
                    }
                }
            }
        }

    return weights;
}
