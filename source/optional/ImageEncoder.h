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
	class ImageEncoder;
	
    /*!
    \brief Image encoder work item. Internal use only.
    */
	class ImageEncoderActivateWorkItem : public WorkItem {
	public:
		ImageEncoder* _pEncoder;

		int _cx, _cy;

		ImageEncoderActivateWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief Image learn work item. Internal use only.
    */
    class ImageEncoderLearnWorkItem : public WorkItem {
    public:
        ImageEncoder* _pEncoder;

        int _cx, _cy;

        float _beta;

        ImageEncoderLearnWorkItem()
            : _pEncoder(nullptr)
        {}

        void run(size_t threadIndex) override;
    };
	
    /*!
    \brief Encoders values to a columnar SDR through random transformation.
    */
    class ImageEncoder {
    private:
        int _inputWidth, _inputHeight;
        int _hiddenWidth, _hiddenHeight;
        int _columnSize;
        int _radius;

        std::vector<int> _hiddenStates;
        std::vector<float> _hiddenActivations;

        std::vector<float> _weights;
        std::vector<float> _biases;

		void activate(int cx, int cy);
		void reconstruct(int cx, int cy);
        void learn(int cx, int cy, float beta);

		std::vector<float> _inputs;
		
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
            unsigned long seed);

        /*!
        \brief Activate the encoder from an input (compute hidden states, perform encoding).
        \param cs compute system to be used.
        \param input input vector/image.
        */
        const std::vector<int> &activate(ComputeSystem &cs, const std::vector<float> &inputs);

        /*!
        \brief Experimental learning functionality.
        \param cs compute system to be used.
        \param beta bias learning rate.
        */
        void learn(ComputeSystem &cs, float beta);

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
		
		friend class ImageEncoderActivateWorkItem;
		friend class ImageEncoderReconstructWorkItem;
        friend class ImageEncoderLearnWorkItem;
    };
}
