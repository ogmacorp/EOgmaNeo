// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <random>
#include <istream>
#include <ostream>
#include <unordered_map>

namespace eogmaneo {
    /*!
    \brief Sigmoid function.
    */
    float sigmoid(float x);

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
    \brief Forward work item, for internal use only.
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
        \brief Radius of sparse weight matrices.
        */
		int _forwardRadius;
        int _backwardRadius;
        //!@}

        /*!
        \brief Whether or not this visible layer should be predicted (used to save processing power).
        */
        bool _predict;

        /*!
        \brief Initialize defaults.
        */
		VisibleLayerDesc()
			: _width(16), _height(16), _chunkSize(4),
			_forwardRadius(6), _backwardRadius(6),
            _predict(true)
		{}
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
        
        std::vector<std::vector<float>> _feedForwardWeights;
        std::vector<std::vector<std::vector<std::pair<float, float>>>> _feedBackWeights;

        std::vector<std::vector<std::pair<float, float>>> _reconActivations;
        std::vector<std::vector<std::pair<float, float>>> _reconActivationsPrev;

        std::vector<VisibleLayerDesc> _visibleLayerDescs;

        std::vector<std::vector<int>> _predictions;
        
        std::vector<std::vector<int>> _inputs;
        std::vector<std::vector<int>> _inputsPrev;
        
        std::vector<int> _feedBack;
        std::vector<int> _feedBackPrev;

        float _alpha;
        float _beta;
  
        void createFromStream(std::istream &s);
        void writeToStream(std::ostream &s);

    public:
        /*!
        \brief Create a layer.
        \param hiddenWidth width of the layer.
        \param hiddenHeight height of the layer.
        \param chunkSize chunk size of the layer.
        \param hasFeedBack whether this layer has feed back.
        \param visibleLayerDescs descriptor structures for all visible layers this (hidden) layer has.
        \param seed random number generator seed for layer generation.
        */
        void create(int hiddenWidth, int hiddenHeight, int chunkSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs, unsigned long seed);

        /*!
        \brief Forward activation and learning.
        \param inputs vector of input SDRs in chunked format.
        \param cs compute system to be used.
        \param alpha feed forward learning rate.
        \param gamma learning decay.
        */
        void forward(const std::vector<std::vector<int> > &inputs, ComputeSystem &cs, float alpha);

        /*!
        \brief Backward activation.
        \param feedBack vector of feedback SDRs in chunked format.
        \param cs compute system to be used.
        \param beta feedback learning rate.
        */
        void backward(const std::vector<int> &feedBack, ComputeSystem &cs, float beta);

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
        \brief Get hidden states, in chunked format.
        */
        const std::vector<int> getHiddenStates() const {
            return _hiddenStates;
        }

        /*!
        \brief Get hidden states, in chunked format.
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
        \brief Get predictions of a visible layer, in chunked format.
        */
        const std::vector<int> getPredictions(int v) const {
            return _predictions[v];
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
        */
        std::vector<float> getFeedBackWeights(int v, int f, int x, int y) const;

        friend class ForwardWorkItem;
        friend class BackwardWorkItem;
        friend class Hierarchy;
    };
}
