// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ComputeSystem.h"

namespace eogmaneo {
    /*!
    \brief Whitener work item. For internal use only.
    */
	class WhitenerWorkItem : public WorkItem {
	public:
		const std::vector<float>* _psrc;
		std::vector<float>* _pdest;

		int _cx, _cy;
		int _width;
		int _radius;
		float _strength;
		int _chunkSize;

		WhitenerWorkItem()
			: _psrc(nullptr), _pdest(nullptr)
		{}

		void run(size_t threadIndex) override;
	};

    /*!
    \brief Sobel X work item. For internal use only.
    */
    class SobelXWorkItem : public WorkItem {
    public:
        const std::vector<float>* _psrc;
        std::vector<float>* _pdest;

        int _cx, _cy;
        int _width;
        int _chunkSize;

        SobelXWorkItem()
            : _psrc(nullptr), _pdest(nullptr)
        {}

        void run(size_t threadIndex) override;
    };
	
    /*!
    \brief Sobel Y work item. For internal use only.
    */
    class SobelYWorkItem : public WorkItem {
    public:
        const std::vector<float>* _psrc;
        std::vector<float>* _pdest;

        int _cx, _cy;
        int _width;
        int _chunkSize;

        SobelYWorkItem()
            : _psrc(nullptr), _pdest(nullptr)
        {}

        void run(size_t threadIndex) override;
    };

    /*!
    \brief Sobel combine work item. For internal use only.
    */
    class SobelCombineWorkItem : public WorkItem {
    public:
        const std::vector<float>* _psrcX;
        const std::vector<float>* _psrcY;
        std::vector<float>* _pdest;

        int _cx, _cy;
        int _width;
        int _chunkSize;

        SobelCombineWorkItem()
            : _psrcX(nullptr), _psrcY(nullptr), _pdest(nullptr)
        {}

        void run(size_t threadIndex) override;
    };

    /*!
    \brief Whiten an image, stored in a raveled vector.
    \param src source image.
    \param width width of the image.
    \param radius radius of the whitening kernel.
    \param strength intensity of the whitening.
    \param cs compute system to be used.
    \param chunkSize diameter of a chunk of computation (for performance only, unlike for the main portion of this library).
    */
    std::vector<float> whiten(const std::vector<float> &src, int width, int radius, float strength, ComputeSystem &cs, int chunkSize);

    void whiten(const std::vector<float> &src, std::vector<float> &dest, int width, int radius, float strength, int cx, int cy, int chunkSize);

    /*!
    \brief Sobel filter (edge detect) an image, stored in a raveled vector.
    \param src source image.
    \param width width of the image.
    \param cs compute system to be used.
    \param chunkSize diameter of a chunk of computation (for performance only, unlike for the main portion of this library).
    */
    std::vector<float> sobel(const std::vector<float> &src, int width, ComputeSystem &cs, int chunkSize);
    
    void sobelX(const std::vector<float> &src, std::vector<float> &dest, int width, int cx, int cy, int chunkSize);
    void sobelY(const std::vector<float> &src, std::vector<float> &dest, int width, int cx, int cy, int chunkSize);
    void sobelCombine(const std::vector<float> &srcX, const std::vector<float> &srcY, std::vector<float> &dest, int width, int cx, int cy, int chunkSize);
}
