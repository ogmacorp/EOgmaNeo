// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "VisAdapter.h"

using namespace eogmaneo;

sf::Packet &eogmaneo::operator << (sf::Packet &packet, const SDR &sdr) {
    packet << sdr._chunkSize << sdr._width << sdr._height;

    for (int i = 0; i < sdr._chunkIndices.size(); i++)
        packet << sdr._chunkIndices[i];

    return packet;
}

sf::Packet &eogmaneo::operator << (sf::Packet &packet, const WeightSet &weightSet) {
    packet << weightSet._name << weightSet._radius;

    sf::Uint16 diam = weightSet._radius * 2 + 1;

    for (int i = 0; i < weightSet._weights.size(); i++)
        packet << weightSet._weights[i];

    return packet;
}

sf::Packet &eogmaneo::operator << (sf::Packet &packet, const Network &network) {
    packet << network._numLayers << network._numWeightSets;

    for (int i = 0; i < network._sdrs.size(); i++)
        packet << network._sdrs[i];

    for (int i = 0; i < network._weightSets.size(); i++)
        packet << network._weightSets[i];

    return packet;
}

sf::Packet &eogmaneo::operator >> (sf::Packet &packet, Caret &caret) {
    return packet >> caret._layer >> caret._bitIndex;
}

void VisAdapter::create(Hierarchy* pHierarchy, int port) {
    _pHierarchy = pHierarchy;

    _listener.listen(port);

    _selector.add(_listener);
}

void VisAdapter::update(float waitSeconds) {
    bool received = false;

    if (_selector.wait(sf::seconds(waitSeconds))) {
        if (_ready && _selector.isReady(_socket)) {
            sf::Packet packet;

            _socket.receive(packet);

            packet >> _caret;

            received = true;
        }
        else if (_selector.isReady(_listener)) {
            if (_listener.accept(_socket) == sf::Socket::Status::Done) {
                // Add to selector
                _selector.add(_socket);

                _ready = true;
            }
        }
    }

    if (_ready) {
        // Send data
        Network network;

        network._numLayers = _pHierarchy->getNumLayers();

        network._sdrs.resize(network._numLayers);

        for (int l = 0; l < network._numLayers; l++) {
            network._sdrs[l]._chunkSize = _pHierarchy->getLayer(l).getChunkSize();
            network._sdrs[l]._width = _pHierarchy->getLayer(l).getHiddenWidth() / _pHierarchy->getLayer(l).getChunkSize();
            network._sdrs[l]._height = _pHierarchy->getLayer(l).getHiddenHeight() / _pHierarchy->getLayer(l).getChunkSize();
            network._sdrs[l]._chunkIndices.resize(_pHierarchy->getLayer(l).getHiddenStates().size());

            for (int i = 0; i < network._sdrs[l]._chunkIndices.size(); i++)
                network._sdrs[l]._chunkIndices[i] = _pHierarchy->getLayer(l).getHiddenStates()[i];
        }

        int caretX = _caret._bitIndex % _pHierarchy->getLayer(_caret._layer).getHiddenWidth();
        int caretY = _caret._bitIndex / _pHierarchy->getLayer(_caret._layer).getHiddenWidth();

        // Visible
        for (int v = 0; v < _pHierarchy->getLayer(_caret._layer).getNumVisibleLayers(); v++) {
            // Feed forward
            {
                WeightSet ws;

                ws._name = "ff_" + std::to_string(v);

                ws._radius = _pHierarchy->getLayer(_caret._layer).getVisibleLayerDesc(v)._radius;

                ws._weights = _pHierarchy->getLayer(_caret._layer).getFeedForwardWeights(v, caretX, caretY);

                network._weightSets.push_back(ws);
            }

            // From hidden state
            if (_pHierarchy->getLayer(_caret._layer).getVisibleLayerDesc(v)._predict) {
                {
                    WeightSet ws;

                    ws._name = "p_h_" + std::to_string(v);

                    ws._weights = _pHierarchy->getLayer(_caret._layer).getPredictionWeights(0, v, caretX, caretY);

                    ws._radius = (std::sqrt(ws._weights.size()) - 1) / 2;

                    network._weightSets.push_back(ws);
                }

                // From feed back
                if (_caret._layer < _pHierarchy->getNumLayers() - 1) {
                    WeightSet ws;

                    ws._name = "p_fb_" + std::to_string(v);

                    ws._weights = _pHierarchy->getLayer(_caret._layer).getPredictionWeights(1, v, caretX, caretY);

                    ws._radius = (std::sqrt(ws._weights.size()) - 1) / 2;

                    network._weightSets.push_back(ws);
                }
            }
        }

        network._numWeightSets = network._weightSets.size();

        sf::Packet packet;

        packet << network;

        _socket.send(packet);
    }
}
