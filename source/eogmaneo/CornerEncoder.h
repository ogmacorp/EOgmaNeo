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
    class CornerEncoder;

    /*!
    \brief Corner encoder work item. For internal use only.
    */
    class CornerEncoderWorkItem : public WorkItem {
    public:
        CornerEncoder* _pEncoder;

        int _cx, _cy;
        bool _useDistanceMetric;

        CornerEncoderWorkItem()
            : _pEncoder(nullptr)
        {}

        void run(size_t threadIndex) override;
    };

    /*!
    \brief FAST-based corner encoder that looks for corners and assigns them to a chunked SDR.
    */
    class CornerEncoder {
    private:
        int _inputWidth, _inputHeight;
        int _chunkSize;
        float _radius;
        int _k;
        float _thresh;
        int _samples;

        std::vector<int> _hiddenScores;

        std::vector<std::vector<int> > _hiddenStates;

        void activate(int cx, int cy);

        std::vector<float> _input;

        std::vector<std::pair<int, int> > _deltas;

    public:
        /*!
        \brief Create the corner encoder.
        \param inputWidth width of the input image. The output (encoded) SDR has the same size as the input.
        \param inputHeight height of the input image. The output (encoded) SDR has the same size as the input.
        \param chunkSize chunk diameter of the output (encoded) chunked SDR.
        \param k number of corners per chunk (multiple SDRs can be generated).
        */
        void create(int inputWidth, int inputHeight, int chunkSize, int k);

        /*!
        \brief Zero the hidden states.
        */
        void clearHiddenStates() {
            for (int order = 0; order < _hiddenStates.size(); order++) {
                int size = _hiddenStates[order].size();

                _hiddenStates[order].clear();
                _hiddenStates[order].assign(size, 0);
            }
        }

        /*!
        \brief Activate (encoded) from an image.
        \param input the raveled input image.
        \param system compute system to be used.
        \param radius radius of the corner detector.
        \param thresh threshold of the corner detector.
        \param samples number of samples around the radius.
        */
        void activate(const std::vector<float> &input, ComputeSystem &system, float radius, float thresh, int samples);

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

        /*!
        \brief Get chunk size of the output SDR.
        */
        int getChunkSize() const {
            return _chunkSize;
        }
    
        /*!
        \brief Get last used radius.
        */
        float getRadius() const {
            return _radius;
        }

        /*!
        \brief Get k
        */
        int getK() const {
            return _k;
        }

        /*!
        \brief Get hidden states at a certain order (order < k).
        Lower indices are more strongly detected corners.
        */
        const std::vector<int> &getHiddenStates(int order) const {
            return _hiddenStates[order];
        }

        friend class CornerEncoderWorkItem;
    };
}