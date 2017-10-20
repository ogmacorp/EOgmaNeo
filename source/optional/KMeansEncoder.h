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
	class KMeansEncoder;
	
    /*!
    \brief K-means encoder work item. Internal use only.
    */
	class KMeansEncoderWorkItem : public WorkItem {
	public:
		KMeansEncoder* _pEncoder;

		int _cx, _cy;

		KMeansEncoderWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief K-means decoder work item. Internal use only.
    */
	class KMeansDecoderWorkItem : public WorkItem {
	public:
		KMeansEncoder* _pEncoder;

		int _cx, _cy;

		KMeansDecoderWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};

    /*!
    \brief K-means learn work item. Internal use only.
    */
    class KMeansLearnWorkItem : public WorkItem {
    public:
        KMeansEncoder* _pEncoder;

        int _cx, _cy;
        float _alpha;
        float _gamma;
        float _minDistance;

        KMeansLearnWorkItem()
            : _pEncoder(nullptr)
        {}

        void run(size_t threadIndex) override;
    };
	
    /*!
    \brief Encoders values to a chunked SDR through linear transformation followed by winner-takes-all.
    */
    class KMeansEncoder {
    private:
        int _inputWidth, _inputHeight;
        int _hiddenWidth, _hiddenHeight;
        int _chunkSize;
        int _radius;

        std::vector<int> _hiddenStates;
        std::vector<int> _hiddenStatesPrev;

        std::vector<float> _hiddenActivations;
        std::vector<float> _hiddenBiases;

        std::vector<float> _weights;
		
		void activate(int cx, int cy);
		void reconstruct(int cx, int cy);
        void learn(int cx, int cy, float alpha, float gamma, float maxDistance);

		std::vector<int> _reconHiddenStates;
		std::vector<float> _input;
		std::vector<float> _recon;
		std::vector<float> _count;
		
    public:
        /*!
        \brief Create the K-means encoder.
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
            float initMinWeight, float initMaxWeight, unsigned long seed);

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
        \param cs compute system to be used.
        */
        const std::vector<int> &activate(const std::vector<float> &input, ComputeSystem &cs);

        /*!
        \brief Reconstruct (reverse) an encoding.
        \param hiddenStates hidden state vector in chunked format.
        \param cs compute system to be used.
        \return reconstructed vector.
        */
        const std::vector<float> &reconstruct(const std::vector<int> &hiddenStates, ComputeSystem &cs);

        /*!
        \brief Learning functionality.
        \param alpha weight learning rate.
        \param gamma bias learning rate.
        \param cs compute system to be used.
        */
        void learn(float alpha, float gamma, float maxDistance, ComputeSystem &cs);

        /*!
        \brief Save to file.
        */
        void save(const std::string &fileName);

        /*1
        \brief Load from file.
        */
        bool load(const std::string &fileName);

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
		
		friend class KMeansEncoderWorkItem;
		friend class KMeansDecoderWorkItem;
        friend class KMeansLearnWorkItem;
    };
}
