// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Agent.h"

#include <algorithm>
#include <fstream>
#include <assert.h>
#include <iostream>

using namespace eogmaneo;

void QForwardWorkItem::run(size_t threadIndex) {
    const Layer* pLayer = &_pAgent->_ph->getLayer(_l);
    QLayer* pQLayer = &_pAgent->_qLayers[_l];

    int hiddenChunkSize = pLayer->getChunkSize();

    int hiddenChunksInX = pLayer->getHiddenWidth() / hiddenChunkSize;
    int hiddenChunksInY = pLayer->getHiddenHeight() / hiddenChunkSize;

    int hiddenChunkX = _hiddenChunkIndex % hiddenChunksInX;
    int hiddenChunkY = _hiddenChunkIndex / hiddenChunksInX;

    // Extract input views
    float activation = 0.0f;
    float count = 0.0f;

    int c = _pAgent->_hiddenStatesTemp[_l][_hiddenChunkIndex];

    int dhx = c % hiddenChunkSize;
    int dhy = c / hiddenChunkSize;

    int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * pLayer->getHiddenWidth();

    if (_l == 0) {
        for (int a = 0; a < _pAgent->_actions.size(); a++) {
            int visibleChunkSize = _pAgent->_actionChunkSizes[a];

            int visibleChunksInX = std::get<0>(_pAgent->_actionSizes[a]) / visibleChunkSize;
            int visibleChunksInY = std::get<1>(_pAgent->_actionSizes[a]) / visibleChunkSize;

            float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
            float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

            int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
            int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

            int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
            int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

            int spatialVisibleRadius = _pAgent->_qLayerDescs[_l]._qRadius;

            int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

            int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

            int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
            int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

            int upperVisibleX = visibleCenterX + spatialVisibleRadius;
            int upperVisibleY = visibleCenterY + spatialVisibleRadius;

            int i = a + _pAgent->_actions.size() * hIndex;
                        
            for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
                for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                    int cx = visibleChunkCenterX + dcx;
                    int cy = visibleChunkCenterY + dcy;

                    if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                        int visibleChunkIndex = cx + cy * visibleChunksInX;

                        int maxIndex = _pAgent->_actionsTemp[a][visibleChunkIndex];

                        int mdx = maxIndex % visibleChunkSize;
                        int mdy = maxIndex / visibleChunkSize;

                        int vx = cx * visibleChunkSize + mdx;
                        int vy = cy * visibleChunkSize + mdy;

                        if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                            int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                            activation += pQLayer->_qWeights[i][wi];
                            count += 1.0f;
                        }
                    }
                }
        }
    }
    else {
        const Layer* pLayerPrev = &_pAgent->_ph->getLayer(_l - 1);
        QLayer* pQLayerPrev = &_pAgent->_qLayers[_l - 1];

        int visibleChunkSize = pLayerPrev->getChunkSize();

        int visibleChunksInX = pLayerPrev->getHiddenWidth() / visibleChunkSize;
        int visibleChunksInY = pLayerPrev->getHiddenHeight() / visibleChunkSize;

        float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
        float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

        int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
        int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

        int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
        int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

        int spatialVisibleRadius = _pAgent->_qLayerDescs[_l]._qRadius;

        int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

        int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

        int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
        int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

        int upperVisibleX = visibleCenterX + spatialVisibleRadius;
        int upperVisibleY = visibleCenterY + spatialVisibleRadius;

        int i = hIndex;
                    
        for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
            for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                int cx = visibleChunkCenterX + dcx;
                int cy = visibleChunkCenterY + dcy;

                if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                    int visibleChunkIndex = cx + cy * visibleChunksInX;

                    int maxIndex = _pAgent->_hiddenStatesTemp[_l - 1][visibleChunkIndex];

                    int mdx = maxIndex % visibleChunkSize;
                    int mdy = maxIndex / visibleChunkSize;

                    int vx = cx * visibleChunkSize + mdx;
                    int vy = cy * visibleChunkSize + mdy;

                    if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                        int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                        activation += pQLayer->_qWeights[i][wi] * pQLayerPrev->_hiddenActivations[visibleChunkIndex];
                        count += 1.0f;
                    }
                }
            }
    }
    
	pQLayer->_hiddenActivations[_hiddenChunkIndex] = activation / std::max(1.0f, count);
}

void QBackwardWorkItem::run(size_t threadIndex) {
    const Layer* pLayer = &_pAgent->_ph->getLayer(_l);
    QLayer* pQLayer = &_pAgent->_qLayers[_l];

    int hiddenChunkSize = pLayer->getChunkSize();

    int hiddenChunksInX = pLayer->getHiddenWidth() / hiddenChunkSize;
    int hiddenChunksInY = pLayer->getHiddenHeight() / hiddenChunkSize;

    int hiddenChunkX = _hiddenChunkIndex % hiddenChunksInX;
    int hiddenChunkY = _hiddenChunkIndex / hiddenChunksInX;

    // Extract input views
    int c = _pAgent->_hiddenStatesTemp[_l][_hiddenChunkIndex];

    int dhx = c % hiddenChunkSize;
    int dhy = c / hiddenChunkSize;

    int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * pLayer->getHiddenWidth();

    float error = pQLayer->_hiddenErrors[_hiddenChunkIndex] / std::max(1.0f, pQLayer->_hiddenCounts[_hiddenChunkIndex]);

    if (_l == 0) {
        for (int a = 0; a < _pAgent->_actions.size(); a++) {
            int visibleChunkSize = _pAgent->_actionChunkSizes[a];

            int visibleBitsPerChunk = visibleChunkSize * visibleChunkSize;

            int visibleChunksInX = std::get<0>(_pAgent->_actionSizes[a]) / visibleChunkSize;
            int visibleChunksInY = std::get<1>(_pAgent->_actionSizes[a]) / visibleChunkSize;

            float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
            float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

            int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
            int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

            int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
            int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

            int spatialVisibleRadius = _pAgent->_qLayerDescs[_l]._qRadius;

            int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

            int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

            int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
            int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

            int upperVisibleX = visibleCenterX + spatialVisibleRadius;
            int upperVisibleY = visibleCenterY + spatialVisibleRadius;

            int i = a + _pAgent->_actions.size() * hIndex;
                        
            for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
                for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                    int cx = visibleChunkCenterX + dcx;
                    int cy = visibleChunkCenterY + dcy;

                    if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                        int visibleChunkIndex = cx + cy * visibleChunksInX;

                        for (int index = 0; index < visibleBitsPerChunk; index++) {
                            int mdx = index % visibleChunkSize;
                            int mdy = index / visibleChunkSize;

                            int vx = cx * visibleChunkSize + mdx;
                            int vy = cy * visibleChunkSize + mdy;

                            if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                                int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                                int vIndex = vx + vy * std::get<0>(_pAgent->_actionSizes[a]);

                                _pAgent->_actionErrors[a][vIndex] += error * pQLayer->_qWeights[i][wi];
                                _pAgent->_actionCounts[a][vIndex] += 1.0f;
                            }
                        }
                    }
                }
        }
    }
    else {
        const Layer* pLayerPrev = &_pAgent->_ph->getLayer(_l - 1);
        QLayer* pQLayerPrev = &_pAgent->_qLayers[_l - 1];

        int visibleChunkSize = pLayerPrev->getChunkSize();

        int visibleChunksInX = pLayerPrev->getHiddenWidth() / visibleChunkSize;
        int visibleChunksInY = pLayerPrev->getHiddenHeight() / visibleChunkSize;

        float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
        float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

        int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
        int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

        int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
        int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

        int spatialVisibleRadius = _pAgent->_qLayerDescs[_l]._qRadius;

        int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

        int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

        int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
        int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

        int upperVisibleX = visibleCenterX + spatialVisibleRadius;
        int upperVisibleY = visibleCenterY + spatialVisibleRadius;

        int i = hIndex;
                    
        for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
            for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                int cx = visibleChunkCenterX + dcx;
                int cy = visibleChunkCenterY + dcy;

                if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                    int visibleChunkIndex = cx + cy * visibleChunksInX;

                    int maxIndex = _pAgent->_hiddenStatesTemp[_l - 1][visibleChunkIndex];

                    int mdx = maxIndex % visibleChunkSize;
                    int mdy = maxIndex / visibleChunkSize;

                    int vx = cx * visibleChunkSize + mdx;
                    int vy = cy * visibleChunkSize + mdy;

                    if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                        int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                        pQLayerPrev->_hiddenErrors[visibleChunkIndex] += error * pQLayer->_qWeights[i][wi];
                        pQLayerPrev->_hiddenCounts[visibleChunkIndex] += 1.0f;
                    }
                }
            }
    }
}

void QLearnWorkItem::run(size_t threadIndex) {
    const Layer* pLayer = &_pAgent->_ph->getLayer(_l);
    QLayer* pQLayer = &_pAgent->_qLayers[_l];

    int hiddenChunkSize = pLayer->getChunkSize();

    int hiddenChunksInX = pLayer->getHiddenWidth() / hiddenChunkSize;
    int hiddenChunksInY = pLayer->getHiddenHeight() / hiddenChunkSize;

    int hiddenChunkX = _hiddenChunkIndex % hiddenChunksInX;
    int hiddenChunkY = _hiddenChunkIndex / hiddenChunksInX;

    // Extract input views
    int c = _pAgent->_hiddenStatesTemp[_l][_hiddenChunkIndex];

    int dhx = c % hiddenChunkSize;
    int dhy = c / hiddenChunkSize;

    int hIndex = (hiddenChunkX * hiddenChunkSize + dhx) + (hiddenChunkY * hiddenChunkSize + dhy) * pLayer->getHiddenWidth();

    float alphaError = _pAgent->_alpha * pQLayer->_hiddenErrors[_hiddenChunkIndex] / std::max(1.0f, pQLayer->_hiddenCounts[_hiddenChunkIndex]);

    if (_l == 0) {
        for (int a = 0; a < _pAgent->_actions.size(); a++) {
            int visibleChunkSize = _pAgent->_actionChunkSizes[a];

            int visibleChunksInX = std::get<0>(_pAgent->_actionSizes[a]) / visibleChunkSize;
            int visibleChunksInY = std::get<1>(_pAgent->_actionSizes[a]) / visibleChunkSize;

            float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
            float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

            int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
            int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

            int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
            int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

            int spatialVisibleRadius = _pAgent->_qLayerDescs[_l]._qRadius;

            int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

            int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

            int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
            int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

            int upperVisibleX = visibleCenterX + spatialVisibleRadius;
            int upperVisibleY = visibleCenterY + spatialVisibleRadius;

            int i = a + _pAgent->_actions.size() * hIndex;
                        
            for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
                for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                    int cx = visibleChunkCenterX + dcx;
                    int cy = visibleChunkCenterY + dcy;

                    if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                        int visibleChunkIndex = cx + cy * visibleChunksInX;

                        int maxIndex = _pAgent->_actionsTemp[a][visibleChunkIndex];

                        int mdx = maxIndex % visibleChunkSize;
                        int mdy = maxIndex / visibleChunkSize;

                        int vx = cx * visibleChunkSize + mdx;
                        int vy = cy * visibleChunkSize + mdy;

                        if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                            int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                            int vIndex = vx + vy * std::get<0>(_pAgent->_actionSizes[a]);

                            pQLayer->_qWeights[i][wi] += alphaError;
                        }
                    }
                }
        }
    }
    else {
        const Layer* pLayerPrev = &_pAgent->_ph->getLayer(_l - 1);
        QLayer* pQLayerPrev = &_pAgent->_qLayers[_l - 1];

        int visibleChunkSize = pLayerPrev->getChunkSize();

        int visibleChunksInX = pLayerPrev->getHiddenWidth() / visibleChunkSize;
        int visibleChunksInY = pLayerPrev->getHiddenHeight() / visibleChunkSize;

        float toInputX = static_cast<float>(visibleChunksInX) / static_cast<float>(hiddenChunksInX);
        float toInputY = static_cast<float>(visibleChunksInY) / static_cast<float>(hiddenChunksInY);

        int visibleChunkCenterX = (hiddenChunkX + 0.5f) * toInputX;
        int visibleChunkCenterY = (hiddenChunkY + 0.5f) * toInputY;

        int visibleCenterX = (visibleChunkCenterX + 0.5f) * visibleChunkSize;
        int visibleCenterY = (visibleChunkCenterY + 0.5f) * visibleChunkSize;

        int spatialVisibleRadius = _pAgent->_qLayerDescs[_l]._qRadius;

        int spatialVisibleDiam = spatialVisibleRadius * 2 + 1;

        int spatialChunkRadius = std::ceil(static_cast<float>(spatialVisibleRadius) / static_cast<float>(visibleChunkSize));

        int lowerVisibleX = visibleCenterX - spatialVisibleRadius;
        int lowerVisibleY = visibleCenterY - spatialVisibleRadius;

        int upperVisibleX = visibleCenterX + spatialVisibleRadius;
        int upperVisibleY = visibleCenterY + spatialVisibleRadius;

        int i = hIndex;
                    
        for (int dcx = -spatialChunkRadius; dcx <= spatialChunkRadius; dcx++)
            for (int dcy = -spatialChunkRadius; dcy <= spatialChunkRadius; dcy++) {
                int cx = visibleChunkCenterX + dcx;
                int cy = visibleChunkCenterY + dcy;

                if (cx >= 0 && cx < visibleChunksInX && cy >= 0 && cy < visibleChunksInY) {
                    int visibleChunkIndex = cx + cy * visibleChunksInX;

                    int maxIndex = _pAgent->_hiddenStatesTemp[_l - 1][visibleChunkIndex];

                    int mdx = maxIndex % visibleChunkSize;
                    int mdy = maxIndex / visibleChunkSize;

                    int vx = cx * visibleChunkSize + mdx;
                    int vy = cy * visibleChunkSize + mdy;

                    if (vx >= lowerVisibleX && vx <= upperVisibleX && vy >= lowerVisibleY && vy <= upperVisibleY) {
                        int wi = (vx - lowerVisibleX) + (vy - lowerVisibleY) * spatialVisibleDiam;

                        pQLayer->_qWeights[i][wi] += alphaError * pQLayerPrev->_hiddenActivations[visibleChunkIndex];
                    }
                }
            }
    }
}

void Agent::qForward(ComputeSystem &cs) {
    for (int l = 0; l < _qLayers.size(); l++) {
        int chunkSize = _ph->getLayer(l).getChunkSize();
        int chunksInX = _ph->getLayer(l).getHiddenWidth() / chunkSize;
        int chunksInY = _ph->getLayer(l).getHiddenHeight() / chunkSize;

        int numChunks = chunksInX * chunksInY;

        std::uniform_int_distribution<int> seedDist(0, 99999);

        // Queue tasks
        for (int i = 0; i < numChunks; i++) {
            std::shared_ptr<QForwardWorkItem> item = std::make_shared<QForwardWorkItem>();

            item->_l = l;
            item->_hiddenChunkIndex = i;
            item->_pAgent = this;
            item->_rng.seed(seedDist(cs._rng));
                
            cs._pool.addItem(item);
        }

        cs._pool.wait();
    }
}

void Agent::qBackward(ComputeSystem &cs) {
    for (int a = 0; a < _actionErrors.size(); a++) {
        _actionErrors[a] = std::vector<float>(_actionErrors[a].size(), 0.0f);
        _actionCounts[a] = std::vector<float>(_actionCounts[a].size(), 0.0f);
    }
    
    // Clear errors for summation
    for (int l = 0; l < _qLayers.size() - 1; l++) {
        _qLayers[l]._hiddenErrors = std::vector<float>(_qLayers[l]._hiddenErrors.size(), 0.0f);
        _qLayers[l]._hiddenCounts = std::vector<float>(_qLayers[l]._hiddenCounts.size(), 0.0f);
    }

    for (int l = _qLayers.size() - 1; l >= 0; l--) {
        int chunkSize = _ph->getLayer(l).getChunkSize();
        int chunksInX = _ph->getLayer(l).getHiddenWidth() / chunkSize;
        int chunksInY = _ph->getLayer(l).getHiddenHeight() / chunkSize;

        int numChunks = chunksInX * chunksInY;

        std::uniform_int_distribution<int> seedDist(0, 99999);

        // Queue tasks
        for (int i = 0; i < numChunks; i++) {
            std::shared_ptr<QBackwardWorkItem> item = std::make_shared<QBackwardWorkItem>();

            item->_l = l;
            item->_hiddenChunkIndex = i;
            item->_pAgent = this;
            item->_rng.seed(seedDist(cs._rng));
                
            cs._pool.addItem(item);
        }

        cs._pool.wait();
    }
}

void Agent::qLearn(ComputeSystem &cs) {
    for (int l = 0; l < _qLayers.size(); l++) {
        int chunkSize = _ph->getLayer(l).getChunkSize();
        int chunksInX = _ph->getLayer(l).getHiddenWidth() / chunkSize;
        int chunksInY = _ph->getLayer(l).getHiddenHeight() / chunkSize;

        int numChunks = chunksInX * chunksInY;

        std::uniform_int_distribution<int> seedDist(0, 99999);

        // Queue tasks
        for (int i = 0; i < numChunks; i++) {
            std::shared_ptr<QLearnWorkItem> item = std::make_shared<QLearnWorkItem>();

            item->_l = l;
            item->_hiddenChunkIndex = i;
            item->_pAgent = this;
            item->_rng.seed(seedDist(cs._rng));
                
            cs._pool.addItem(item);
        }

        cs._pool.wait();
    }
}

void Agent::create(Hierarchy* ph, const std::vector<std::pair<int, int> > &actionSizes, const std::vector<int> &actionChunkSizes, const std::vector<QLayerDesc> &qLayerDescs, unsigned long seed) {
    std::mt19937 rng(seed);

    _ph = ph;

    _actionSizes = actionSizes;
    _actionChunkSizes = actionChunkSizes;

    _qLayerDescs = qLayerDescs;

    _qLayers.resize(qLayerDescs.size());

    assert(_qLayers.size() == ph->getNumLayers());

    std::uniform_real_distribution<float> initWeightDistQ(0.999f, 1.001f);

    _actions.resize(actionSizes.size());
    _actionErrors.resize(actionSizes.size());
    _actionCounts.resize(actionSizes.size());

    for (int a = 0; a < _actions.size(); a++) {
        _actions[a].resize((std::get<0>(actionSizes[a]) / actionChunkSizes[a]) * (std::get<1>(actionSizes[a]) / actionChunkSizes[a]), 0);
        _actionErrors[a].resize(std::get<0>(actionSizes[a]) * std::get<1>(actionSizes[a]), 0.0f);
        _actionCounts[a].resize(std::get<0>(actionSizes[a]) * std::get<1>(actionSizes[a]), 0.0f);
    }

    for (int l = 0; l < _qLayers.size(); l++) {
        int numHidden = ph->getLayer(l).getHiddenWidth() * ph->getLayer(l).getHiddenHeight();
        int numChunks = (ph->getLayer(l).getHiddenWidth() / ph->getLayer(l).getChunkSize()) * (ph->getLayer(l).getHiddenHeight() / ph->getLayer(l).getChunkSize());

        _qLayers[l]._hiddenActivations.resize(numChunks, 0.0f);
        _qLayers[l]._hiddenErrors.resize(numChunks, 0.0f);
        _qLayers[l]._hiddenCounts.resize(numChunks, 0.0f);

        if (l == 0) {
            _qLayers[l]._qWeights.resize(numHidden * _actions.size());

            int qVecSize = qLayerDescs[l]._qRadius * 2 + 1;

            qVecSize *= qVecSize;

            for (int i = 0; i < _qLayers[l]._qWeights.size(); i++) {
                _qLayers[l]._qWeights[i].resize(qVecSize);

                for (int j = 0; j < qVecSize; j++)
                    _qLayers[l]._qWeights[i][j] = initWeightDistQ(rng);
            }
        }
        else {
            _qLayers[l]._qWeights.resize(numHidden);

            int qVecSize = qLayerDescs[l]._qRadius * 2 + 1;

            qVecSize *= qVecSize;

            for (int i = 0; i < _qLayers[l]._qWeights.size(); i++) {
                _qLayers[l]._qWeights[i].resize(qVecSize);

                for (int j = 0; j < qVecSize; j++)
                    _qLayers[l]._qWeights[i][j] = initWeightDistQ(rng);
            }
        }
    }
}

void Agent::step(const std::vector<std::vector<int> > &inputs, ComputeSystem &cs, float reward, bool learn) {
    _ph->step(inputs, cs, learn);

    // Capture hidden state
    std::vector<std::vector<int> > hiddenStates(_ph->getNumLayers());

    for (int l = 0; l < hiddenStates.size(); l++)
        hiddenStates[l] = _ph->getLayer(l).getHiddenStates();

    // Find best action
    _hiddenStatesTemp = hiddenStates;

    _qLayers.back()._hiddenErrors = std::vector<float>(_qLayers.back()._hiddenErrors.size(), 1.0f);
    _qLayers.back()._hiddenCounts = std::vector<float>(_qLayers.back()._hiddenCounts.size(), 1.0f);

    qBackward(cs);

    // Find highest bits
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int a = 0; a < _actions.size(); a++) {
        int actionBits = _actionChunkSizes[a] * _actionChunkSizes[a];

        for (int i = 0; i < _actions[a].size(); i++) {
            if (dist01(cs._rng) < _epsilon) {
                std::uniform_int_distribution<int> chunkDist(0, actionBits - 1);

                _actions[a][i] = chunkDist(cs._rng);
            }
            else {
                int index = 0;

                for (int j = 1; j < actionBits; j++)
                    if (_actionErrors[a][i * actionBits + j] / std::max(1.0f, _actionCounts[a][i * actionBits + j]) > _actionErrors[a][i * actionBits + index] / std::max(1.0f, _actionCounts[a][i * actionBits + index]))
                        index = j;

                _actions[a][i] = index;
            }
        }
    }

    // Find Q values
    _actionsTemp = _actions;

    qForward(cs);

    // Get state sample
    HistorySample hs;

    hs._actions = _actions;
    hs._hiddenStates = hiddenStates;
    hs._reward = 0.0f;

    // Updates
    if (learn && !_historySamples.empty()) {
        std::uniform_int_distribution<int> sampleDist(0, _historySamples.size() - 1);

        float nextQ = 0.0f;

        for (int i = 0; i < _qLayers.back()._hiddenActivations.size(); i++)
            nextQ += _qLayers.back()._hiddenActivations[i];

        nextQ /= _qLayers.back()._hiddenActivations.size();
        
        std::vector<float> qTargets(_historySamples.size());

        _historySamples.front()._reward = reward;

        for (int t = 0; t < qTargets.size(); t++) {
            float q = (1.0f - _gamma) * _historySamples[t]._reward + _gamma * nextQ;
                
            qTargets[t] = q;

            nextQ = q;
        }

        for (int iter = 0; iter < _sampleIter; iter++) {
            int index = sampleDist(cs._rng);

            // Activate
            _hiddenStatesTemp = _historySamples[index]._hiddenStates;
            _actionsTemp = _historySamples[index]._actions;

            qForward(cs);

            for (int i = 0; i < _qLayers.back()._hiddenActivations.size(); i++) {
                _qLayers.back()._hiddenErrors[i] = qTargets[index] - _qLayers.back()._hiddenActivations[i];
                _qLayers.back()._hiddenCounts[i] = 1.0f;
            }

            qBackward(cs);

            qLearn(cs);
        }
    }

    _historySamples.insert(_historySamples.begin(), hs);

    if (_historySamples.size() > _maxHistorySamples)
        _historySamples.resize(_maxHistorySamples);
}