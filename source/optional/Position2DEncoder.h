// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <random>

namespace eogmaneo {
    /*!
    \brief Encoder 2D position into a single chunk.
    */
    class Position2DEncoder {
    private:
        int _chunkSize;
    
        int _seed;

        float _scale;
		
    public:
        /*!
        \brief Create the Position2D encoder.
        */
        void create(int chunkSize, float scale, unsigned long seed);

        /*!
        \brief Activate the encoder from a position.
        */
        int activate(const std::pair<float, float> &position, ComputeSystem &cs);
    };
}
