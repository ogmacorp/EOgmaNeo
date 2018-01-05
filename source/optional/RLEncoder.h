// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <unordered_map>
#include <random>

namespace eogmaneo {
	class RLEncoder;
	
    /*!
    \brief Image encoder work item. Internal use only.
    */
	class RLEncoderCrossWorkItem : public WorkItem {
	public:
		RLEncoder* _pEncoder;

		int _cx, _cy;

        float _reward, _alpha, _gamma, _epsilon, _tau, _traceDecay, _minTrace;

		RLEncoderCrossWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief Image decoder work item. Internal use only.
    */
	class RLEncoderActionWorkItem : public WorkItem {
	public:
		RLEncoder* _pEncoder;

		int _cx, _cy;

        float _reward, _alpha, _gamma, _epsilon, _tau, _traceDecay, _minTrace;

		RLEncoderActionWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};

    /*!
    \brief Encoders values to a chunked SDR through random transformation.
    */
    class RLEncoder {
    private:
        int _actionWidth, _actionHeight;
        int _hiddenWidth, _hiddenHeight;
        int _actionChunkSize, _hiddenChunkSize;
        int _actionRadius, _crossRadius;

        std::vector<int> _actions;
        std::vector<int> _hiddenStates;

        std::vector<int> _predictions;

        std::vector<float> _actionQsPrev;
        std::vector<float> _actionMaxQsPrev;
        std::vector<float> _hiddenQsPrev;
        std::vector<float> _hiddenMaxQsPrev;

        std::vector<std::vector<float>> _crossWeights;
		std::vector<std::vector<float>> _actionWeights;
		
        std::vector<std::unordered_map<int, float>> _crossTraces;
        std::vector<std::unordered_map<int, float>> _actionTraces;

        std::mt19937 _rng;

		void updateCross(int cx, int cy, float reward, float alpha, float gamma, float epsilon, float tau, float traceDecay, float minTrace);
		void updateAction(int cx, int cy, float reward, float alpha, float gamma, float epsilon, float tau, float traceDecay, float minTrace);

    public:
        /*!
        \brief Create the random encoder.
        */
        void create(int actionWidth, int actionHeight, int actionChunkSize, int hiddenWidth, int hiddenHeight, int hiddenChunkSize, int actionRadius, int crossRadius,
            unsigned long seed);

        /*!
        \brief Step the encoder.
        \param predictions prediction SDR.
        \param cs compute system to be used.
        */
        void step(const std::vector<int> &predictions, ComputeSystem &cs, float reward, float alpha, float gamma, float epsilon, float tau, float traceDecay, float minTrace = 0.01f);

        //!@{
        /*!
        \brief Get action dimensions.
        */
        int getActionWidth() const {
            return _actionWidth;
        }

        int getActionHeight() const {
            return _actionHeight;
        }
        //!@}

        //!@{
        /*!
        \brief Get hidden dimensions.
        */
        int getHiddenWidth() const {
            return _hiddenWidth;
        }

        int getHiddenHeight() const {
            return _hiddenHeight;
        }
        //!@}

        /*!
        \brief Get (hidden) chunk size.
        */
        int getHiddenChunkSize() const {
            return _hiddenChunkSize;
        }

        /*!
        \brief Get (hidden) chunk size.
        */
        int getActionChunkSize() const {
            return _actionChunkSize;
        }

        /*!
        \brief Get radius of weights onto the hidden state.
        */
        int getCrossRadius() const {
            return _crossRadius;
        }

        /*!
        \brief Get radius of weights onto the input.
        */
        int getActionRadius() const {
            return _actionRadius;
        }

        /*!
        \brief Get lastly computed hidden states.
        */
        const std::vector<int> &getActions() const {
            return _actions;
        }

        /*!
        \brief Get lastly computed hidden states.
        */
        const std::vector<int> &getHiddenStates() const {
            return _hiddenStates;
        }
		
		friend class RLEncoderCrossWorkItem;
		friend class RLEncoderActionWorkItem;
    };
}
