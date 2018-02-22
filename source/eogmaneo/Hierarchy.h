// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Layer.h"
#include "ComputeSystem.h"

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
        \brief The size of a chunk.
        This size is the diameter of the chunk. The number of bits in a chunk is therefore _chunkSize^2.
        */
		int _chunkSize;

        //!@{
        /*!
        \brief Radii of forward and backward sparse weight matrices.
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
        \brief Encoder learning rate.
        */
        float _alpha;

        /*!
        \brief Decoder learning rate.
        */
        float _beta;

        /*!
        \brief Initialize defaults.
        */
		LayerDesc()
			: _width(24), _height(24), _chunkSize(6),
			_forwardRadius(9), _backwardRadius(9),
			_ticksPerUpdate(2), _temporalHorizon(2),
			_alpha(0.1f), _beta(0.1f)
		{}
	};

    /*!
    \brief A hierarchy of layers, or agent (if reward is supplied).
    */
    class Hierarchy {
    private:
        std::vector<Layer> _layers;

        std::vector<std::vector<std::vector<int> > > _histories;

        std::vector<float> _alphas;
        std::vector<float> _betas;

        std::vector<bool> _updates;

        std::vector<int> _ticks;
        std::vector<int> _ticksPerUpdate;

        int _inputTemporalHorizon;
        int _numInputs;

    public:
        /*!
        \brief Create the hierarchy.
        \param inputSizes vector of input dimension tuples.
        \param inputChunkSizes vector of input chunk sizes (diameters).
        \param predictInputs flags for which inputs to generate predictions for.
        \param layerDescs vector of LayerDesc structures, describing each layer in sequence.
        \param seed random number generator seed for generating the hierarchy.
        */
        void create(const std::vector<std::pair<int, int> > &inputSizes, const std::vector<int> &inputChunkSizes, const std::vector<bool> &predictInputs, const std::vector<LayerDesc> &layerDescs, unsigned long seed);

        /*!
        \brief Load a hierarchy from a file instead of creating it randomly (as done by create(...) ).
        Takes the file name.
        */
        bool load(const std::string &fileName);

        /*!
        \brief Save a hierarchy to a file.
        Takes the file name.
        */
        void save(const std::string &fileName);

        /*!
        \brief Simulation tick.
        \param inputs vector of SDR vectors in chunked format.
        \param cs compute system to be used.
        \param learn whether learning should be enabled, defaults to true.
        \param topFeedBack SDR vector in chunked format of top-level feedback state.
        */
        void step(const std::vector<std::vector<int> > &inputs, ComputeSystem &cs, bool learn = true, const std::vector<int> &topFeedBack = {});

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

            return _layers.front()._predictions[index];
        }
        
        //!@{
        /*!
        \brief Accessors for layer parameters.
        */
        float getAlpha(int l) const {
            return _alphas[l];
        }

        float getBeta(int l) const {
            return _betas[l];
        }
        //!@}

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
        const Layer &getLayer(int l) const {
            return _layers[l];
        }
    };
}
