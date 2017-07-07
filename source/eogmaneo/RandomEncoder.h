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

namespace eogmaneo {
	class RandomEncoder;
	
    /*!
    \brief Random encoder work item. Internal use only.
    */
	class RandomEncoderWorkItem : public WorkItem {
	public:
		RandomEncoder* _pEncoder;

		int _cx, _cy;
		bool _useDistanceMetric;

		RandomEncoderWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief Random decoder work item. Internal use only.
    */
	class RandomDecoderWorkItem : public WorkItem {
	public:
		RandomEncoder* _pEncoder;

		int _cx, _cy;

		RandomDecoderWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};

    /*!
    \brief Random learn work item. Internal use only.
    */
    class RandomLearnWorkItem : public WorkItem {
    public:
        RandomEncoder* _pEncoder;

        int _cx, _cy;
        float _alpha;
        float _gamma;

        RandomLearnWorkItem()
            : _pEncoder(nullptr)
        {}

        void run(size_t threadIndex) override;
    };
	
    /*!
    \brief Encoders values to a chunked SDR through random transformation.
    */
    class RandomEncoder {
    private:
        int _inputWidth, _inputHeight;
        int _hiddenWidth, _hiddenHeight;
        int _chunkSize;
        int _radius;

        std::vector<int> _hiddenStates;

        std::vector<float> _hiddenActivations;
        std::vector<float> _hiddenBiases;

        std::vector<float> _weights;
		
		void activate(int cx, int cy, bool useDistanceMetric);
		void reconstruct(int cx, int cy);
        void learn(int cx, int cy, float alpha, float gamma);

		std::vector<int> _reconHiddenStates;
		std::vector<float> _input;
		std::vector<float> _recon;
		std::vector<float> _count;
		
    public:
        /*!
        \brief Create the random encoder.
        \param inputWidth input image width.
        \param inputHeight input image height.
        \param hiddenWidth hidden SDR width.
        \param hiddenHeight hidden SDR height.
        \param chunkSize chunk diameter of hidden SDR.
        \param radius radius onto the input.
        \param initMinWeight initial smallest weight (random, uniform).
        \param initMaxWeight initial largest weight (random, uniform).
        \param seed random number generator seed used when generating this encoder.
        \param normalize whether to normalize (L2) the weights.
        */
        void create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int chunkSize, int radius,
            float initMinWeight, float initMaxWeight, unsigned long seed, bool normalize);

        /*!
        \brief Zero the hidden states.
        */
        void clearHiddenStates() {
            int size = _hiddenStates.size();

            _hiddenStates.clear();
            _hiddenStates.assign(size, 0);
        }

        /*!
        \brief Activate the encoder from an input (compute hidden states, perform encoding).
        \param input input vector/image.
        \param system compute system to be used.
        \param useDistanceMetric whether to activate based on euclidean distance (true) or dot product (false). Defaults to true.
        */
        const std::vector<int> &activate(const std::vector<float> &input, System &system, bool useDistanceMetric = true);

        /*!
        \brief Reconstruct (reverse) an encoding.
        \param hiddenStates hidden state vector in chunked format.
        \param system compute system to be used.
        \return reconstructed vector.
        */
        const std::vector<float> &reconstruct(const std::vector<int> &hiddenStates, System &system);

        /*!
        \brief Experimental learning functionality.
        Requires that reconstruct(...) has been called, without another call to activate(...).
        \param alpha weight learning rate.
        \param gamma bias learning rate.
        \param system compute system to be used.
        */
        void learn(float alpha, float gamma, System &system);

        //!@{
        /*!
        \brief Get input dimensions.
        */
        int getInputWidth() const {
            return _inputWidth;
        }

        int getInputHeight() const {
            return _inputHeight;
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
        int getChunkSize() const {
            return _chunkSize;
        }

        /*!
        \brief Get radius of weights onto the input.
        */
        int getRadius() const {
            return _radius;
        }

        /*!
        \brief Get lastly computed hidden states.
        */
        const std::vector<int> &getHiddenStates() const {
            return _hiddenStates;
        }
		
		friend class RandomEncoderWorkItem;
		friend class RandomDecoderWorkItem;
        friend class RandomLearnWorkItem;
    };
}
