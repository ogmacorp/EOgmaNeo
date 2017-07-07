// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "System.h"

#include <random>
#include <istream>
#include <ostream>

namespace eogmaneo {
    /*!
    \brief Forward work item, for internal use only.
    */
    class ForwardWorkItem : public WorkItem {
    public:
        class Layer* _pLayer;

        int _hiddenChunkIndex;

        std::mt19937 _rng;

        ForwardWorkItem()
            : _pLayer(nullptr)
        {}

        void run(size_t threadIndex) override;
    };

    /*!
    \brief Backward work item, for internal use only.
    */
    class BackwardWorkItem : public WorkItem {
    public:
        class Layer* _pLayer;

        int _visibleChunkIndex;
        int _visibleLayerIndex;
        
        std::mt19937 _rng;

        BackwardWorkItem()
            : _pLayer(nullptr)
        {}

        void run(size_t threadIndex) override;
    };

    /*!
    \brief Visible layer parameters.
    Describes a visible (input) layer.
    */
	struct VisibleLayerDesc {
        //!@{
        /*!
        \brief Visible layer dimensions (2D).
        */
		int _width, _height;
        //!@}
        
        /*!
        \brief Chunk size of the input.
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
        \brief Whether this layer is predicted (has a backward pass). Only ever false for input layers and overflowing (temporalHorizon > ticksPerUpdate) visible layers.
        */
        bool _predict;

        /*!
        \brief Initialize defaults.
        */
		VisibleLayerDesc()
			: _width(36), _height(36), _chunkSize(6),
			_forwardRadius(9), _backwardRadius(9),
			_predict(true)
		{}
	};
    
    /*!
    \brief A replay sample. For internal use.
    */
    struct ReplaySample {
        std::vector<std::vector<int> > _predictionsPrev;

        std::vector<std::vector<int> > _feedBackPrev;

        float _reward;
    };
	
    /*!
    \brief A layer in the hierarchy.
    */
    class Layer {
    private:
        int _hiddenWidth;
        int _hiddenHeight;
        int _chunkSize;

        std::vector<int> _hiddenStates;
        std::vector<int> _hiddenStatesPrev;
        
        std::vector<std::vector<float> > _feedForwardWeights;

        std::vector<VisibleLayerDesc> _visibleLayerDescs;

        std::vector<std::vector<int>> _predictions;
        std::vector<std::vector<int>> _predictionsPrev;

        std::vector<std::vector<std::vector<float> > > _feedBackWeights;

        std::vector<std::vector<int>> _inputs;
        std::vector<std::vector<int>> _inputsPrev;

        std::vector<std::vector<int>> _feedBack;
        std::vector<std::vector<int>> _feedBackPrev;

        float _alpha;
        float _beta;
        float _delta;
        float _gamma;
        float _epsilon;
        float _reward;
        
        std::vector<ReplaySample> _replaySamples;

        void createFromStream(std::istream &s);
        void writeToStream(std::ostream &s);

    public:
        /*!
        \brief Maximum number of samples stored for replay.
        */
        int _maxReplaySamples;

        /*!
        \brief Iterations over replay buffer per layer tick.
        */
        int _replayIter;
        
        /*!
        \brief Initialize defaults.
        */
        Layer()
            : _maxReplaySamples(100), _replayIter(4)
        {}
        
        /*!
        \brief Create a layer.
        \param hiddenWidth width of the layer.
        \param hiddenHeight height of the layer.
        \param chunkSize chunk size of the layer.
        \param hasFeedBack whether or not this layer receives feedback from a higher layer.
        \param visibleLayerDescs descriptor structures for all visible layers this (hidden) layer has.
        \param seed random number generator seed for layer generation.
        */
        void create(int hiddenWidth, int hiddenHeight, int chunkSize, bool hasFeedBack, const std::vector<VisibleLayerDesc> &visibleLayerDescs, unsigned long seed);

        /*!
        \brief Forward activation.
        \param inputs vector of input SDRs in chunked format.
        \param system compute system to be used.
        \param alpha feed forward learning rate.
        */
        void forward(const std::vector<std::vector<int> > &inputs, System &system, float alpha);

        /*!
        \brief Backward activation.
        \param feedBack vector of feedback SDRs in chunked format.
        \param system compute system to be used.
        \param reward reinforcement signal.
        \param beta feedback learning rate.
        \param delta Q learning rate.
        \param gamma Q discount factor.
        \param epsilon Q exploration rate.
        */
        void backward(const std::vector<std::vector<int> > &feedBack, System &system, float reward, float beta, float delta, float gamma, float epsilon);

        //!@{
        /*!
        \brief Get dimensions.
        */
        int getHiddenWidth() const {
            return _hiddenWidth;
        }

        int getHiddenHeight() const {
            return _hiddenHeight;
        }
        //!@}

        /*!
        \brief Get the chunk size.
        */
        int getChunkSize() const {
            return _chunkSize;
        }

        /*!
        \brief Get the number of visible layers this (hidden) layer uses.
        */
        int getNumVisibleLayers() const {
            return _visibleLayerDescs.size();
        }

        /*!
        \brief Retrieve a visible layer descriptor.
        */
        const VisibleLayerDesc &getVisibleLayerDesc(int v) const {
            return _visibleLayerDescs[v];
        }

        /*!
        \brief Get the number of feedback layers. Usually 1 or 2 (no feedback / with feedback).
        */
        int getNumFeedBackLayers() const {
            return _feedBack.size();
        }

        /*!
        \brief Get hidden states, in chunked format.
        */
        const std::vector<int> getHiddenStates() const {
            return _hiddenStates;
        }

        /*!
        \brief Get previous timestep hidden states, in chunked format.
        */
        const std::vector<int> getHiddenStatesPrev() const {
            return _hiddenStatesPrev;
        }

        /*!
        \brief Get inputs of a visible layer, in chunked format.
        */
        const std::vector<int> getInputs(int v) const {
            return _inputs[v];
        }

        /*!
        \brief Get previous timestep inputs of a visible layer, in chunked format.
        */
        const std::vector<int> getInputsPrev(int v) const {
            return _inputsPrev[v];
        }

        /*!
        \brief Get predictions of a visible layer, in chunked format.
        */
        const std::vector<int> getPredictions(int v) const {
            return _predictions[v];
        }

        /*!
        \brief Get previous timestep predictions of a visible layer, in chunked format.
        */
        const std::vector<int> getPredictionsPrev(int v) const {
            return _predictionsPrev[v];
        }

        /*!
        \brief Get feedback layer, in chunked format.
        */
        const std::vector<int> getFeedBack(int f) const {
            return _feedBack[f];
        }

        /*!
        \brief Get previous timestep feedback layer, in chunked format.
        */
        const std::vector<int> getFeedBackPrev(int f) const {
            return _feedBackPrev[f];
        }

        /*!
        \brief Get feedforward weights of a particular unit.
        */
        const std::vector<float> &getFeedForwardWeights(int v, int x, int y) const {
            int i = v + _visibleLayerDescs.size() * (x + y * _hiddenWidth);

            return _feedForwardWeights[i];
        }

        /*!
        \brief Get feedback weights of a particular unit.
        This function is expensive, it reprojects weights such that they are addressed from the current layer instead of the visible layer.
        */
        std::vector<float> getFeedBackWeights(int v, int f, int x, int y) const;

        friend class ForwardWorkItem;
        friend class BackwardWorkItem;
        friend class Hierarchy;
    };
}
