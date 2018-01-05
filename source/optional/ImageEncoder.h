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
	class ImageEncoder;
	
    /*!
    \brief Image encoder work item. Internal use only.
    */
	class ImageEncoderWorkItem : public WorkItem {
	public:
		ImageEncoder* _pEncoder;

		int _cx, _cy;

		ImageEncoderWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};
	
    /*!
    \brief Image decoder work item. Internal use only.
    */
	class ImageDecoderWorkItem : public WorkItem {
	public:
		ImageEncoder* _pEncoder;

		int _cx, _cy;

		ImageDecoderWorkItem()
			: _pEncoder(nullptr)
		{}

		void run(size_t threadIndex) override;
	};

    /*!
    \brief Image learn work item. Internal use only.
    */
    class ImageLearnWorkItem : public WorkItem {
    public:
        ImageEncoder* _pEncoder;

        int _cx, _cy;

        float _alpha;

        ImageLearnWorkItem()
            : _pEncoder(nullptr)
        {}

        void run(size_t threadIndex) override;
    };
	
    /*!
    \brief Encoders values to a chunked SDR through random transformation.
    */
    class ImageEncoder {
    private:
        int _inputWidth, _inputHeight;
        int _hiddenWidth, _hiddenHeight;
        int _chunkSize;
        int _radius;

        std::vector<int> _hiddenStates;

        std::vector<float> _hiddenActivations;

        std::vector<float> _weights;
		
		void activate(int cx, int cy);
		void reconstruct(int cx, int cy);
        void learn(int cx, int cy, float alpha);

		std::vector<int> _reconHiddenStates;
		std::vector<float> _input;
		std::vector<float> _recon;
		std::vector<float> _count;

        std::vector<std::vector<float>> _samples;
		
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
        \brief Add a training sample.
        */
        void addSample(const std::vector<float> &input, int maxSamples = 100);

        /*!
        \brief Experimental learning functionality.
        Requires that reconstruct(...) has been called, without another call to activate(...).
        \param alpha weight learning rate.
        \param cs compute system to be used.
        */
        void learn(float alpha, ComputeSystem &cs, int iter = 5);

        /*!
        \brief Save to a file.
        */
        void save(const std::string &fileName);

        /*!
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
		
		friend class ImageEncoderWorkItem;
		friend class ImageDecoderWorkItem;
        friend class ImageLearnWorkItem;
    };
}
