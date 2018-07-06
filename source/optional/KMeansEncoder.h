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

namespace eogmaneo {
	class KMeansEncoder;
	
    /*!
    \brief Image encoder work item. Internal use only.
    */
	class KMeansEncoderActivateWorkItem : public WorkItem {
	public:
		KMeansEncoder* _pEncoder;

		int _cx, _cy;

		KMeansEncoderActivateWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief Image decoder work item. Internal use only.
    */
	class KMeansEncoderReconstructWorkItem : public WorkItem {
	public:
		KMeansEncoder* _pEncoder;

		int _cx, _cy;

		KMeansEncoderReconstructWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};

    /*!
    \brief Image learn work item. Internal use only.
    */
    class KMeansEncoderLearnWorkItem : public WorkItem {
    public:
        KMeansEncoder* _pEncoder;

        int _cx, _cy;

        float _alpha;

        KMeansEncoderLearnWorkItem()
            : _pEncoder(nullptr)
        {}

        void run(size_t threadIndex) override;
    };
	
    /*!
    \brief Encoders values to a columnar SDR through random transformation.
    */
    class KMeansEncoder {
    private:
        int _inputWidth, _inputHeight;
        int _hiddenWidth, _hiddenHeight;
        int _columnSize;
        int _radius;

        std::vector<int> _hiddenStates;

        std::vector<float> _weights;

		void activate(int cx, int cy);
		void reconstruct(int cx, int cy);
        void learn(int cx, int cy, float alpha);

		std::vector<int> _reconHiddenStates;
		std::vector<float> _inputs;
		std::vector<float> _recons;
		std::vector<float> _counts;
		
    public:
        /*!
        \brief Create the random encoder.
        \param inputWidth input image width.
        \param inputHeight input image height.
        \param hiddenWidth hidden SDR width.
        \param hiddenHeight hidden SDR height.
        \param columnSize column size of hidden SDR.
        \param radius radius onto the input.
        \param seed random number generator seed used when generating this encoder.
        */
        void create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int columnSize, int radius,
            float initMinWeight, float initMaxWeight, 
            unsigned long seed);

        /*!
        \brief Activate the encoder from an input (compute hidden states, perform encoding).
        \param input input vector/image.
        \param cs compute system to be used.
        */
        const std::vector<int> &activate(ComputeSystem &cs, const std::vector<float> &inputs);

        /*!
        \brief Reconstruct (reverse) an encoding.
        \param hiddenStates hidden state vector in columnar format.
        \param cs compute system to be used.
        \return reconstructed vector.
        */
        const std::vector<float> &reconstruct(ComputeSystem &cs, const std::vector<int> &hiddenStates);

        /*!
        \brief Experimental learning functionality.
        Requires that reconstruct(...) has been called, without another call to activate(...).
        \param alpha weight learning rate.
        \param cs compute system to be used.
        */
        void learn(ComputeSystem &cs, float alpha);

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
        int getColumnSize() const {
            return _columnSize;
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
		
		friend class KMeansEncoderActivateWorkItem;
		friend class KMeansEncoderReconstructWorkItem;
        friend class KMeansEncoderLearnWorkItem;
    };
}
