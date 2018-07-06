// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Hierarchy.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <assert.h>

using namespace eogmaneo;

void Hierarchy::create(const std::vector<std::pair<int, int> > &inputSizes, const std::vector<int> &inputColumnSizes, const std::vector<bool> &predictInputs, const std::vector<LayerDesc> &layerDescs, unsigned long seed) {
    std::mt19937 rng(seed);

    _layers.resize(layerDescs.size());

    _ticks.resize(layerDescs.size(), 0);

    _histories.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    _updates.resize(layerDescs.size(), false);

	_inputTemporalHorizon = layerDescs.front()._temporalHorizon;
    _inputSizes = inputSizes;

    for (int l = 0; l < layerDescs.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l]._ticksPerUpdate; // First layer always 1

    for (int l = 0; l < layerDescs.size(); l++) {
        _histories[l].resize(l == 0 ? _inputSizes.size() * layerDescs[l]._temporalHorizon : layerDescs[l]._temporalHorizon);

        std::vector<VisibleLayerDesc> visibleLayerDescs;

        if (l == 0) {
            visibleLayerDescs.resize(_inputSizes.size() * layerDescs[l]._temporalHorizon);

            for (int i = 0; i < _inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = i * layerDescs[l]._temporalHorizon + t;

                    visibleLayerDescs[index]._width = std::get<0>(inputSizes[i]);
                    visibleLayerDescs[index]._height = std::get<1>(inputSizes[i]);
                    visibleLayerDescs[index]._columnSize = inputColumnSizes[i];
                    visibleLayerDescs[index]._forwardRadius = layerDescs[l]._forwardRadius;
                    visibleLayerDescs[index]._backwardRadius = layerDescs[l]._backwardRadius;
                    visibleLayerDescs[index]._predict = t == 0 && predictInputs[i];
                }
            }
			
			for (int v = 0; v < _histories[l].size(); v++) {
				int in = v / layerDescs[l]._temporalHorizon;
				
				_histories[l][v].resize(std::get<0>(inputSizes[in]) * std::get<1>(inputSizes[in]), 0);	
			}
        }
        else {
            visibleLayerDescs.resize(layerDescs[l]._temporalHorizon);

            for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                visibleLayerDescs[t]._width = layerDescs[l - 1]._width;
                visibleLayerDescs[t]._height = layerDescs[l - 1]._height;
                visibleLayerDescs[t]._columnSize = layerDescs[l - 1]._columnSize;
                visibleLayerDescs[t]._forwardRadius = layerDescs[l]._forwardRadius;
                visibleLayerDescs[t]._backwardRadius = layerDescs[l]._backwardRadius;
                visibleLayerDescs[t]._predict = t < _ticksPerUpdate[l];
            }
			
			for (int v = 0; v < _histories[l].size(); v++)
				_histories[l][v].resize(layerDescs[l - 1]._width * layerDescs[l - 1]._height, 0);
        }
		
        _layers[l].create(layerDescs[l]._width, layerDescs[l]._height, layerDescs[l]._columnSize, visibleLayerDescs, seed + l + 1);
    }
}

void Hierarchy::step(ComputeSystem &cs, const std::vector<std::vector<int>> &inputs, bool learn, const std::vector<int> &topFeedBack) {
    assert(inputs.size() == _inputSizes.size());

    _ticks[0] = 0;

    // Add to first history   
    {
        int temporalHorizon = _histories.front().size() / inputs.size();

        for (int t = temporalHorizon - 1; t > 0; t--) {
            for (int in = 0; in < inputs.size(); in++)
                _histories.front()[t + temporalHorizon * in] = _histories.front()[(t - 1) + temporalHorizon * in];     
        }

        for (int in = 0; in < inputs.size(); in++)
            _histories.front()[0 + temporalHorizon * in] = inputs[in];
    }

    std::vector<int> updates(_layers.size(), false);

    for (int l = 0; l < _layers.size(); l++) {
        if (l == 0 || _ticks[l] >= _ticksPerUpdate[l]) {
            _ticks[l] = 0;

            updates[l] = true;
            
            _layers[l].forward(cs, _histories[l], learn);

            // Add to next layer's history
            if (l < _layers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = _histories[lNext].size();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    _histories[lNext][t] = _histories[lNext][t - 1];

                _histories[lNext].front() = _layers[l].getHiddenStates();

                _ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = _layers.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            std::vector<int> feedBack;

            if (l < _layers.size() - 1)
                feedBack = _layers[l + 1].getPredictions(_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]);
            else
                feedBack = topFeedBack;

            _layers[l].backward(cs, feedBack, learn);
        }
    }

    _updates = updates;
}

void Hierarchy::save(const std::string &fileName) {
    std::ofstream os(fileName, std::ios::binary);

    int numLayers = _layers.size();

    os.write(reinterpret_cast<char*>(&numLayers), sizeof(int));
    os.write(reinterpret_cast<char*>(&_inputTemporalHorizon), sizeof(int));

    int numInputs = _inputSizes.size();

    os.write(reinterpret_cast<char*>(&numInputs), sizeof(int));
    os.write(reinterpret_cast<char*>(_inputSizes.data()), _inputSizes.size() * sizeof(std::pair<int, int>));
    
    // Write additional per-layer data
    os.write(reinterpret_cast<char*>(_ticks.data()), _ticks.size() * sizeof(int));
    os.write(reinterpret_cast<char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));
    os.write(reinterpret_cast<char*>(_updates.data()), _updates.size() * sizeof(int));

    for (int l = 0; l < _layers.size(); l++) {
        int temporalHorizon = l == 0 ? _histories[l].size() / _inputSizes.size() : _histories[l].size();
    
        os.write(reinterpret_cast<char*>(&temporalHorizon), sizeof(int));

        // History
        for (int v = 0; v < _histories[l].size(); v++)
            os.write(reinterpret_cast<char*>(_histories[l][v].data()), _histories[l][v].size() * sizeof(int));

        // Write layer
        _layers[l].writeToStream(os);
    }
}

bool Hierarchy::load(const std::string &fileName) {
    std::ifstream is(fileName, std::ios::binary);

    if (!is.is_open())
        return false;

    int numLayers;

    is.read(reinterpret_cast<char*>(&numLayers), sizeof(int));
    is.read(reinterpret_cast<char*>(&_inputTemporalHorizon), sizeof(int));

    int numInputs;

    is.read(reinterpret_cast<char*>(&numInputs), sizeof(int));

    _inputSizes.resize(numInputs);

    is.read(reinterpret_cast<char*>(_inputSizes.data()), _inputSizes.size() * sizeof(std::pair<int, int>));
    
    _layers.resize(numLayers);

    _ticks.resize(_layers.size());

    _histories.resize(_layers.size());
    
    _ticksPerUpdate.resize(_layers.size());

    _updates.resize(_layers.size());

    // Read additional per-layer data
    is.read(reinterpret_cast<char*>(_ticks.data()), _ticks.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_ticksPerUpdate.data()), _ticksPerUpdate.size() * sizeof(int));
    is.read(reinterpret_cast<char*>(_updates.data()), _updates.size() * sizeof(int));

    for (int l = 0; l < _layers.size(); l++) {
        int temporalHorizon;

        is.read(reinterpret_cast<char*>(&temporalHorizon), sizeof(int));
    
        // History
        _histories[l].resize(l == 0 ? _inputSizes.size() * temporalHorizon : temporalHorizon);

        if (l == 0) {
            for (int v = 0; v < _histories[l].size(); v++) {
                int in = v / temporalHorizon;
                    
                _histories[l][v].resize(std::get<0>(_inputSizes[in]) * std::get<1>(_inputSizes[in]));
                
                is.read(reinterpret_cast<char*>(_histories[l][v].data()), _histories[l][v].size() * sizeof(int));
            }
        }
        else {
            for (int v = 0; v < _histories[l].size(); v++) {
				_histories[l][v].resize(_layers[l - 1].getHiddenWidth() * _layers[l - 1].getHiddenHeight());

                is.read(reinterpret_cast<char*>(_histories[l][v].data()), _histories[l][v].size() * sizeof(int));
            }
        }

        // Read layer
        _layers[l].readFromStream(is);
    }

    return true;
}