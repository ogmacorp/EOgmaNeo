// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Layer.h"

namespace eogmaneo {
    /*!
    \brief Parameters for a layer.
    Used during construction of a hierarchy.
    */
	struct LayerDesc {
        //!@{
        /*!
        \brief Dimensions (2D) of the layer.
        */
		int _width, _height;
        //!@}

        /*!
        \brief The size of a column.
        */
		int _columnSize;

        //!@{
        /*!
        \brief Radii of forward, lateral and backward sparse weight matrices.
        */
		int _forwardRadius;
        int _backwardRadius;
        //!@}

        /*!
        \brief Number of ticks a layer takes to update (relative to previous layer).
        */
		int _ticksPerUpdate;

        /*!
        \brief Temporal distance into a the past addressed by the layer. Should be greater than or equal to _ticksPerUpdate.
        */
		int _temporalHorizon;

        /*!
        \brief Initialize defaults.
        */
		LayerDesc()
			: _width(4), _height(4), _columnSize(16),
			_forwardRadius(2), _backwardRadius(2),
			_ticksPerUpdate(2), _temporalHorizon(2)
		{}
	};

    /*!
    \brief A hierarchy of layers, using exponential memory structure.
    */
    class Hierarchy {
    private:
        std::vector<Layer> _layers;

        std::vector<std::vector<std::vector<int> > > _histories;

        std::vector<int> _updates;

        std::vector<int> _ticks;
        std::vector<int> _ticksPerUpdate;

        int _inputTemporalHorizon;
        std::vector<std::pair<int, int> > _inputSizes;

    public:
        /*!
        \brief Create the hierarchy.
        \param inputSizes vector of input dimension tuples.
        \param inputColumnSizes vector of input column sizes.
        \param predictInputs flags for which inputs to generate predictions for.
        \param layerDescs vector of LayerDesc structures, describing each layer in sequence.
        \param seed random number generator seed for generating the hierarchy.
        */
        void create(const std::vector<std::pair<int, int> > &inputSizes, const std::vector<int> &inputColumnSizes, const std::vector<bool> &predictInputs, const std::vector<LayerDesc> &layerDescs, unsigned long seed);

        /*!
        \brief Simulation step/tick.
        \param cs compute system to be used.
        \param inputs vector of SDR vectors in columnar format.
        \param learn whether learning should be enabled, defaults to true.
        \param topFeedBack SDR vector in columnar format of top-level feed back state.
        */
        void step(ComputeSystem &cs, const std::vector<std::vector<int> > &inputs, bool learn = true, const std::vector<int> &topFeedBack = {});

        /*!
        \brief Save the hierarchy to a file.
        */
        void save(const std::string &fileName);

        /*!
        \brief Load the hierarchy from a file.
        */
        bool load(const std::string &fileName);

        /*!
        \brief Get the number of (hidden) layers.
        */
        int getNumLayers() const {
            return _layers.size();
        }

        /*!
        \brief Get the predicted version of the input.
        \param i the index of the input to retrieve.
        */
        const std::vector<int> &getPredictions(int i) const {
            int index = i * _inputTemporalHorizon;

            return _layers.front().getPredictions(index);
        }

        /*!
        \brief Whether this layer received on update this timestep.
        */
        bool getUpdate(int l) const {
            return _updates[l];
        }

        /*!
        \brief Get current layer ticks, relative to previous layer.
        */
        int getTicks(int l) const {
            return _ticks[l];
        }

        /*!
        \brief Get layer ticks per update, relative to previous layer.
        */
        int getTicksPerUpdate(int l) const {
            return _ticksPerUpdate[l];
        }

        /*!
        \brief Get history of a layer's input.
        */
        const std::vector<std::vector<int> > getHistories(int l) {
            return _histories[l];
        }

        /*!
        \brief Retrieve a layer.
        */
        Layer &getLayer(int l) {
            return _layers[l];
        }
    };
}
