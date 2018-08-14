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

		void run() override;
	};
	
    /*!
    \brief Image encoder work item. Internal use only.
    */
	class ImageEncoderReconstructWorkItem : public WorkItem {
	public:
		ImageEncoder* _pEncoder;

		int _cx, _cy;

		ImageEncoderReconstructWorkItem()
			: _pEncoder(nullptr)
		{}

		void run() override;
	};

    /*!
    \brief Image learn work item. Internal use only.
    */
    class ImageEncoderLearnWorkItem : public WorkItem {
    public:
        ImageEncoder* _pEncoder;

        int _cx, _cy;

        float _alpha;
        float _beta;

        ImageEncoderLearnWorkItem()
            : _pEncoder(nullptr)
        {}

        void run() override;
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
        std::vector<int> _reconHiddenStates;
        std::vector<float> _hiddenActivations;

        std::vector<float> _weightsFF;
        std::vector<float> _weightsR;
        std::vector<float> _biases;

		void activate(int cx, int cy);
		void reconstruct(int cx, int cy);
        void learn(int cx, int cy, float alpha, float beta);

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
            unsigned long seed);

        /*!
        \brief Activate the encoder from an input (compute hidden states, perform encoding).
        \param cs compute system to be used.
        \param input input vector/image.
        \return hidden SDR
        */
        const std::vector<int> &activate(ComputeSystem &cs, const std::vector<float> &inputs);

        /*!
        \brief Reconstruction (reversal).
        \param cs compute system to be used.
        \param reconHiddenStates hidden states to reconstruct.
        \return reconstruction
        */
        const std::vector<float> &reconstruct(ComputeSystem &cs, const std::vector<int> &reconHiddenStates);

        /*!
        \brief Learning.
        \param cs compute system to be used.
        \param alpha bias learning rate.
        \param beta reconstruction learning rate.
        */
        void learn(ComputeSystem &cs, float alpha, float beta);

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

        /*!
        \brief Get lastly computed reconstuction.
        */
        const std::vector<float> &getRecons() const {
            return _recons;
        }
		
		friend class ImageEncoderActivateWorkItem;
		friend class ImageEncoderReconstructWorkItem;
        friend class ImageEncoderLearnWorkItem;
    };
}
