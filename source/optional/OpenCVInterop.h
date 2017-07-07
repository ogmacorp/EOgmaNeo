// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

namespace eogmaneo {
    class OpenCVInterop
    {
    public:
        /* Edge detection methods */

        int CannyEdgeDetection(
            std::vector<float>& data,
            float threshold1, float threshold2,
            int apertureSize = 3, bool L2gradient = false);


        /* Thresholding methods */

        enum
        {
            CV_THRESH_BINARY = 0,       /**< value = value > threshold ? max_value : 0       */
            CV_THRESH_BINARY_INV = 1,   /**< value = value > threshold ? 0 : max_value       */
            CV_THRESH_TRUNC = 2,        /**< value = value > threshold ? threshold : value   */
            CV_THRESH_TOZERO = 3,       /**< value = value > threshold ? value : 0           */
            CV_THRESH_TOZERO_INV = 4,   /**< value = value > threshold ? 0 : value           */
            CV_THRESH_MASK = 7,

            CV_THRESH_OTSU = 8,         /**< use Otsu algorithm to choose the optimal threshold value;
                                            combine the flag with one of the above CV_THRESH_* values */
            CV_THRESH_TRIANGLE = 16     /**< use Triangle algorithm to choose the optimal threshold value;
                                            combine the flag with one of the above CV_THRESH_* values, but not
                                            with CV_THRESH_OTSU */
        };
        enum
        {
            CV_ADAPTIVE_THRESH_MEAN_C = 0,
            CV_ADAPTIVE_THRESH_GAUSSIAN_C = 1
        };

        int Threshold(
            std::vector<float>& data,
            float threshold, float maxValue,
            int type);

        int AdaptiveThreshold(
            std::vector<float>& data,
            float maxValue, int adaptiveMethod, int thresholdType,
            int blockSize, float C);


        /* Gabor filtering */

        int GaborFilter(
            std::vector<float>& data,
            int kernelSize, float sigma, float theta, float lambd, float gamma, float psi);


        /* Line Segment Detection */
        void LineSegmentDetector(
            std::vector<float>& data, int width, int height, int chunkSize,
            std::vector<int>& rotationSDR, bool drawLines = false);
    };
}
