// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

#include <random>

namespace eogmaneo {
	class SparseImageEncoder;
	
    /*!
    \brief Image encoder work item. Internal use only.
    */
	class SparseImageEncoderWorkItem : public WorkItem {
	public:
		SparseImageEncoder* _pEncoder;

		int _cx, _cy;
        int _inputChunkSize;

		SparseImageEncoderWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief Image decoder work item. Internal use only.
    */
	class SparseImageInhibitWorkItem : public WorkItem {
	public:
		SparseImageEncoder* _pEncoder;

		int _cx, _cy;

		SparseImageInhibitWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief Encoders values to a chunked SDR through random transformation.
    */
    class SparseImageEncoder {
    private:
        int _inputWidth, _inputHeight;
        int _hiddenWidth, _hiddenHeight;
        int _chunkSize;
        int _radius;

        std::vector<int> _hiddenStates;
        std::vector<float> _hiddenActivations;

        std::vector<float> _weights;

		void activate(int cx, int cy, int inputChunkSize);
		void inhibit(int cx, int cy);

		std::vector<float> _input;
		
    public:
        /*!
        \brief Create the random encoder.
        \param inputWidth input image width.
        \param inputHeight input image height.
        \param hiddenWidth hidden SDR width.
        \param hiddenHeight hidden SDR height.
        \param chunkSize chunk diameter of hidden SDR.
        \param radius radius onto the input.
        \param seed random number generator seed used when generating this encoder.
        */
        void create(int inputWidth, int inputHeight, int hiddenWidth, int hiddenHeight, int chunkSize, int radius,
            unsigned long seed);

        /*!
        \brief Activate the encoder from an input (compute hidden states, perform encoding).
        \param input input vector/image.
        \param cs compute system to be used.
        */
        const std::vector<int> &activate(const std::vector<float> &input, ComputeSystem &cs, int inputChunkSize = 8);

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
		
		friend class SparseImageEncoderWorkItem;
		friend class SparseImageInhibitWorkItem;
    };
}
