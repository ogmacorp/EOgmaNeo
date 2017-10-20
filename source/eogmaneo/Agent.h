// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Hierarchy.h"

namespace eogmaneo {
    /*!
    \brief Parameters for a Q layer.
    Used during construction of an agent.
    */
	struct QLayerDesc {
        /*!
        \brief Q radius
        */
        int _qRadius;

        /*!
        \brief Initialize defaults.
        */
		QLayerDesc()
			: _qRadius(12)
		{}
	};

    /*!
    \brief Q layer.
    */
    struct QLayer {
        std::vector<std::vector<float> > _qWeights;

        std::vector<float> _hiddenActivations;

        std::vector<float> _hiddenErrors;
        std::vector<float> _hiddenCounts;
    };

    /*!
    \brief Forward work item, for internal use only.
    */
    class QForwardWorkItem : public WorkItem {
    public:
        class Agent* _pAgent;

        int _l;
        int _hiddenChunkIndex;

        std::mt19937 _rng;

        QForwardWorkItem()
            : _pAgent(nullptr)
        {}

        void run(size_t threadIndex) override;
    };

    /*!
    \brief Backward work item, for internal use only.
    */
    class QBackwardWorkItem : public WorkItem {
    public:
        class Agent* _pAgent;

        int _l;
        int _hiddenChunkIndex;

        std::mt19937 _rng;

        QBackwardWorkItem()
            : _pAgent(nullptr)
        {}

        void run(size_t threadIndex) override;
    };

    /*!
    \brief Learn work item, for internal use only.
    */
    class QLearnWorkItem : public WorkItem {
    public:
        class Agent* _pAgent;

        int _l;
        int _hiddenChunkIndex;

        std::mt19937 _rng;

        QLearnWorkItem()
            : _pAgent(nullptr)
        {}

        void run(size_t threadIndex) override;
    };

    /*!
    \brief History sample.
    */
    struct HistorySample {
        std::vector<std::vector<int> > _actions;
        std::vector<std::vector<int> > _hiddenStates;

        float _reward;
    };

    /*!
    \brief An agent, attached to a hierarchy.
    */
    class Agent {
    private:
        Hierarchy* _ph;

        std::vector<HistorySample> _historySamples;

        std::vector<std::pair<int, int> > _actionSizes;
        std::vector<int> _actionChunkSizes;

        std::vector<std::vector<float> > _actionErrors;
        std::vector<std::vector<float> > _actionCounts;

        std::vector<std::vector<int> > _actions;

        std::vector<QLayerDesc> _qLayerDescs;
        std::vector<QLayer> _qLayers;

        std::vector<std::vector<int> > _hiddenStatesTemp;
        std::vector<std::vector<int> > _actionsTemp;

        void qForward(ComputeSystem &cs);
        void qBackward(ComputeSystem &cs);
        void qLearn(ComputeSystem &cs);

    public:
        int _maxHistorySamples;
        int _sampleIter;
        float _alpha;
        float _gamma;
        float _epsilon;

        /*!
        \brief Initialize defaults.
        */
        Agent()
        : _maxHistorySamples(60), _sampleIter(5),
        _alpha(0.01f), _gamma(0.95f), _epsilon(0.02f)
        {}

        /*!
        \brief Create the agent.
        \param qLayerDescs vector of QLayerDesc structures, describing each Q layer in sequence.
        \param seed random number generator seed for generating the hierarchy.
        */
        void create(Hierarchy* ph, const std::vector<std::pair<int, int> > &actionSizes, const std::vector<int> &actionChunkSizes, const std::vector<QLayerDesc> &qLayerDescs, unsigned long seed);

        /*!
        \brief Simulation tick.
        \param inputs vector of SDR vectors in chunked format.
        \param cs compute system to be used.
        \param reward reinforcement signal, defaults to 0.
        \param learn whether learning should be enabled, defaults to true.
        */
        void step(const std::vector<std::vector<int> > &inputs, ComputeSystem &cs, float reward, bool learn = true);

        const std::vector<int> &getActions(int i) const {
            return _actions[i];
        }

        friend class QForwardWorkItem;
        friend class QBackwardWorkItem;
        friend class QLearnWorkItem;
    };
}
