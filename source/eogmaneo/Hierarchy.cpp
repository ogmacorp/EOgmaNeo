// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
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

void Hierarchy::create(const std::vector<std::pair<int, int> > &inputSizes, const std::vector<int> &inputChunkSizes, const std::vector<bool> &predictInputs, const std::vector<LayerDesc> &layerDescs, unsigned long seed) {
    std::mt19937 rng(seed);

    _layers.resize(layerDescs.size());

    _ticks.assign(layerDescs.size(), 0);

    _histories.resize(layerDescs.size());
    
    _ticksPerUpdate.resize(layerDescs.size());

    _alphas.resize(layerDescs.size());
    _betas.resize(layerDescs.size());

    _updates.resize(layerDescs.size(), false);

	_inputTemporalHorizon = layerDescs.front()._temporalHorizon;
    _numInputs = inputSizes.size();

    for (int l = 0; l < layerDescs.size(); l++)
        _ticksPerUpdate[l] = l == 0 ? 1 : layerDescs[l]._ticksPerUpdate; // First layer always 1

    for (int l = 0; l < layerDescs.size(); l++) {
        _histories[l].resize(l == 0 ? inputSizes.size() * layerDescs[l]._temporalHorizon : layerDescs[l]._temporalHorizon);

		_alphas[l] = layerDescs[l]._alpha;
        _betas[l] = layerDescs[l]._beta;

        std::vector<VisibleLayerDesc> visibleLayerDescs;

        if (l == 0) {
            visibleLayerDescs.resize(inputSizes.size() * layerDescs[l]._temporalHorizon);

            for (int i = 0; i < inputSizes.size(); i++) {
                for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                    int index = i * layerDescs[l]._temporalHorizon + t;

                    visibleLayerDescs[index]._width = std::get<0>(inputSizes[i]);
                    visibleLayerDescs[index]._height = std::get<1>(inputSizes[i]);
                    visibleLayerDescs[index]._chunkSize = inputChunkSizes[i];
                    visibleLayerDescs[index]._forwardRadius = layerDescs[l]._forwardRadius;
                    visibleLayerDescs[index]._backwardRadius = layerDescs[l]._backwardRadius;
                    visibleLayerDescs[index]._predict = t == 0 && predictInputs[i];
                }
            }
			
			for (int v = 0; v < _histories[l].size(); v++) {
				int t = v % layerDescs[l]._temporalHorizon;
				int in = v / layerDescs[l]._temporalHorizon;
				
				_histories[l][v].resize((std::get<0>(inputSizes[in]) / inputChunkSizes[in]) * (std::get<1>(inputSizes[in]) / inputChunkSizes[in]), 0);	
			}
        }
        else {
            visibleLayerDescs.resize(layerDescs[l]._temporalHorizon);

            for (int t = 0; t < layerDescs[l]._temporalHorizon; t++) {
                visibleLayerDescs[t]._width = layerDescs[l - 1]._width;
                visibleLayerDescs[t]._height = layerDescs[l - 1]._height;
                visibleLayerDescs[t]._chunkSize = layerDescs[l - 1]._chunkSize;
                visibleLayerDescs[t]._forwardRadius = layerDescs[l]._forwardRadius;
                visibleLayerDescs[t]._backwardRadius = layerDescs[l]._backwardRadius;
                visibleLayerDescs[t]._predict = t < _ticksPerUpdate[l];
            }
			
			for (int v = 0; v < _histories[l].size(); v++)
				_histories[l][v].resize((layerDescs[l - 1]._width / layerDescs[l - 1]._chunkSize) * (layerDescs[l - 1]._height / layerDescs[l - 1]._chunkSize), 0);
        }
		
        _layers[l].create(layerDescs[l]._width, layerDescs[l]._height, layerDescs[l]._chunkSize, visibleLayerDescs, seed + l + 1);
    }
}

bool Hierarchy::load(const std::string &fileName) {
    std::ifstream s(fileName);

    if (!s.is_open())
        return false;

    int numLayers;
    s >> numLayers;

    _layers.resize(numLayers);

    _ticks.assign(numLayers, 0);

    _histories.resize(numLayers);

    _ticksPerUpdate.resize(numLayers);

	_alphas.resize(numLayers);
    _betas.resize(numLayers);

    _updates.resize(numLayers);

	s >> _inputTemporalHorizon;
    s >> _numInputs;

    std::vector<std::pair<int, int>> inputSizes(_numInputs);
    std::vector<int> inputChunkSizes(_numInputs);

    for (int i = 0; i < _numInputs; i++) {
        int w, h, c;

        s >> w >> h >> c;

        inputSizes[i] = std::make_pair(w, h);
        inputChunkSizes[i] = c;
    }

    for (int l = 0; l < numLayers; l++) {
        int temporalHorizon, ticksPerUpdate;
        s >> temporalHorizon >> ticksPerUpdate;

        _histories[l].resize(l == 0 ? inputSizes.size() * temporalHorizon : temporalHorizon);

        _ticksPerUpdate[l] = l == 0 ? 1 : ticksPerUpdate; // First layer always 1

        s >> _alphas[l] >> _betas[l];

        int update;
        s >> update;

        _updates[l] = update;

        _layers[l].createFromStream(s);

        for (int v = 0; v < _histories[l].size(); v++) {
            int w = _layers[l].getVisibleLayerDesc(v)._width;
            int h = _layers[l].getVisibleLayerDesc(v)._height;
            int c = _layers[l].getVisibleLayerDesc(v)._chunkSize;
                
            _histories[l][v].resize((w / c) * (h / c), 0);

            for (int i = 0; i < _histories[l][v].size(); i++)
                s >> _histories[l][v][i];
        }
    }
    
    return true;
}

void Hierarchy::save(const std::string &fileName) {
    std::ofstream s(fileName);

    s << _layers.size() << std::endl;

	s << _inputTemporalHorizon << std::endl;

    s << _numInputs << std::endl;

    for (int i = 0; i < _numInputs; i++) {
        int w, h, c;

        int index = i * _inputTemporalHorizon;

        s << _layers.front().getVisibleLayerDesc(index)._width << " " << _layers.front().getVisibleLayerDesc(index)._height << " " << _layers.front().getVisibleLayerDesc(index)._chunkSize << std::endl;
    }

    for (int l = 0; l < _layers.size(); l++) {
        int temporalHorizon = l == 0 ? (_histories[l].size() / _numInputs) : _histories[l].size();

        s << temporalHorizon << " " << _ticksPerUpdate[l] << std::endl;

        s << _alphas[l] << " " << _betas[l] << std::endl;

        s << (_updates[l] ? 1 : 0) << std::endl;

        _layers[l].writeToStream(s);

        for (int v = 0; v < _histories[l].size(); v++) {
            for (int i = 0; i < _histories[l][v].size(); i++)
                s << _histories[l][v][i] << " ";

            s << std::endl;
        }
    }
}

void Hierarchy::step(const std::vector<std::vector<int>> &inputs, ComputeSystem &cs, bool learn, const std::vector<int> &topFeedBack) {
    assert(inputs.size() == _numInputs);

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

    std::vector<bool> updates(_layers.size(), false);

    for (int l = 0; l < _layers.size(); l++) {
        if (l == 0 || _ticks[l] >= _ticksPerUpdate[l]) {
            _ticks[l] = 0;

            updates[l] = true;
            
            _layers[l].forward(_histories[l], cs, learn ? _alphas[l] : 0.0f);

            // Add to next layer's history
            if (l < _layers.size() - 1) {
                int lNext = l + 1;

                int temporalHorizon = _histories[lNext].size();

                for (int t = temporalHorizon - 1; t > 0; t--)
                    _histories[lNext][t] = _histories[lNext][t - 1];

                _histories[lNext].front() = _layers[l]._hiddenStates;

                _ticks[lNext]++;
            }
        }
    }

    // Backward
    for (int l = _layers.size() - 1; l >= 0; l--) {
        if (updates[l]) {
            std::vector<int> feedBack;

            if (l < _layers.size() - 1)
                feedBack = _layers[l + 1]._predictions[_ticksPerUpdate[l + 1] - 1 - _ticks[l + 1]];
            else {
                if (topFeedBack.size() > 0)
                    feedBack = topFeedBack;
                else
                    feedBack = _layers[l]._hiddenStates;
            }

            _layers[l].backward(feedBack, cs, learn ? _betas[l] : 0.0f);
        }
    }

    _updates = updates;
}
