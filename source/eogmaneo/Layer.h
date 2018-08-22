// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <istream>
#include <ostream>
#include <unordered_map>

namespace eogmaneo {
    /*!
    \brief Sigmoid function.
    */
    float sigmoid(float x);

    class Layer;

    /*!
    \brief Layer forward work item. Internal use only.
    */
	class LayerForwardWorkItem : public WorkItem {
	public:
		Layer* _pLayer;

		int _ci;

		LayerForwardWorkItem()
			: _pLayer(nullptr)
		{}

		void run() override;
	};

    /*!
    \brief Layer backward work item. Internal use only.
    */
	class LayerBackwardWorkItem : public WorkItem {
	public:
		Layer* _pLayer;

		int _ci;
        int _v;

        std::mt19937 _rng;

		LayerBackwardWorkItem()
			: _pLayer(nullptr)
		{}

		void run() override;
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
        \brief Column size of the input.
        This size is the height of a column in each position.
        */
		int _columnSize;

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
			: _width(4), _height(4), _columnSize(16),
			_forwardRadius(2), _backwardRadius(2),
            _predict(true)
		{}
	};

    /*!
    \brief History sample.
    */
    struct HistorySample {
        std::vector<int> _hiddenStates;
        std::vector<int> _feedBack;
        std::vector<std::vector<int> > _inputs;
        float _reward;
    };

    /*!
    \brief A layer in the hierarchy.
    */
    class Layer {
    private:
        int _hiddenWidth;
        int _hiddenHeight;
        int _columnSize;

        std::vector<int> _hiddenStates;
        std::vector<int> _hiddenStatesPrev;
        
        std::vector<std::vector<std::vector<float>>> _feedForwardWeights;
        std::vector<std::vector<std::vector<float>>> _feedBackWeights;

        std::vector<VisibleLayerDesc> _visibleLayerDescs;

        std::vector<std::vector<int>> _predictions;
        
        std::vector<std::vector<int>> _inputs;
        std::vector<std::vector<int>> _inputsPrev;
        
        std::vector<int> _feedBack;

        bool _learn;
        
        std::vector<HistorySample> _historySamples;
  
        void columnForward(int ci);
        void columnBackward(int ci, int v, std::mt19937 &rng);

        /*!
        \brief Write to stream
        */
        void readFromStream(std::istream &is);
        void writeToStream(std::ostream &os);

    public:
        /*!
        \brief Learning rate for feed forward weights.
        */
        float _alpha;
        
        /*!
        \brief Learning rate for feed back weights.
        */
        float _beta;

        /*!
        \brief Discount factor.
        */
        float _gamma;

        /*!
        \brief Maximum number of history samples.
        */
        int _maxHistorySamples;

        /*!
        \brief Initialize defaults.
        */
        Layer()
        : _alpha(0.1f), _beta(0.01f), _gamma(0.9f), _maxHistorySamples(32)
        {}

        /*!
        \brief Create a layer.
        \param hiddenWidth width of the layer.
        \param hiddenHeight height of the layer.
        \param columnSize column size of the layer.
        \param hasFeedBack whether this layer has feed back.
        \param visibleLayerDescs descriptor structures for all visible layers this (hidden) layer has.
        \param seed random number generator seed for layer generation.
        */
        void create(int hiddenWidth, int hiddenHeight, int columnSize, const std::vector<VisibleLayerDesc> &visibleLayerDescs, unsigned long seed);

        /*!
        \brief Forward activation and learning.
        \param inputs vector of input SDRs in chunked format.
        \param learn whether to learn.
        */
        void forward(ComputeSystem &cs, const std::vector<std::vector<int> > &inputs, bool learn);

        /*!
        \brief Backward activation.
        \param feedBack vector of feedback SDRs in chunked format.
        \param reward reinforcement signal.
        \param learn whether to learn.
        */
        void backward(ComputeSystem &cs, const std::vector<int> &feedBack, float reward, bool learn);

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
        int getColumnSize() const {
            return _columnSize;
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
        const std::vector<int> &getHiddenStates() const {
            return _hiddenStates;
        }

        /*!
        \brief Get hidden states, in chunked format.
        */
        const std::vector<int> &getHiddenStatesPrev() const {
            return _hiddenStatesPrev;
        }

        /*!
        \brief Get inputs of a visible layer, in chunked format.
        */
        const std::vector<int> &getInputs(int v) const {
            return _inputs[v];
        }

        /*!
        \brief Get predictions of a visible layer, in chunked format.
        */
        const std::vector<int> &getPredictions(int v) const {
            return _predictions[v];
        }

        friend class LayerForwardWorkItem;
        friend class LayerBackwardWorkItem;

        friend class Hierarchy;
    };
}
