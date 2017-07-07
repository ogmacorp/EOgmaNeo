// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <SFML/System.hpp>
#include <SFML/Network.hpp>

#include "Layer.h"
#include "Hierarchy.h"

#include <thread>

namespace eogmaneo {
    /*!
    \brief VisAdapter SDR. For internal use.
    */
    struct SDR {
        sf::Uint16 _chunkSize;
        sf::Uint16 _width, _height;
        std::vector<sf::Uint16> _chunkIndices;
    };

    sf::Packet &operator << (sf::Packet &packet, const SDR &sdr);

    /*!
    \brief VisAdapter weight set. For internal use.
    */
    struct WeightSet {
        std::string _name;
        sf::Uint16 _radius;
        std::vector<float> _weights;
    };

    sf::Packet &operator << (sf::Packet &packet, const WeightSet &weightSet);

    /*!
    \brief VisAdapter network. For internal use.
    */
    struct Network {
        sf::Uint16 _numLayers;
        sf::Uint16 _numWeightSets;
        std::vector<SDR> _sdrs;
        std::vector<WeightSet> _weightSets;
    };

    sf::Packet &operator << (sf::Packet &packet, const Network &network);

    /*!
    \brief VisAdapter Caret. For internal use.
    */
    struct Caret {
        sf::Uint16 _layer;
        sf::Uint16 _bitIndex;
    };

    sf::Packet &operator >> (sf::Packet &packet, Caret &caret);

    /*!
    \brief Adapter for visualizing a hierarchy using NeoVis.
    */
    class VisAdapter {
    private:
        bool _ready;
        Caret _caret;
        Hierarchy* _pHierarchy;

        sf::SocketSelector _selector;
        sf::TcpListener _listener;
        sf::TcpSocket _socket;

    public:
        VisAdapter()
            : _pHierarchy(nullptr), _ready(false)
        {
            _caret._layer = 0;
            _caret._bitIndex = 0;
        }

        /*!
        \brief Attached to a hierachy.
        \param pHierarchy hierarchy to attach to.
        \param port to communicate to NeoVis over.
        */
        void create(Hierarchy* pHierarchy, int port);

        /*!
        \brief Update NeoVis with current state of the hierarchy.
        \param waitSeconds minimal wait time for new caret position.
        */
        void update(float waitSeconds = 0.001f);
    };
}