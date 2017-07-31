// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#define _USE_MATH_DEFINES
#include <math.h>

#include "OpenCVInterop.h"
#include "opencv2/imgproc.hpp"


int eogmaneo::OpenCVInterop::CannyEdgeDetection(
    std::vector<float>& data,
    float threshold1, float threshold2,
    int apertureSize, bool L2gradient)
{
    // data contains float values [0.0 .. 1.0]
    cv::Mat dataMat(data, false);

    cv::Mat img_in;
    dataMat.convertTo(img_in, CV_8U, 255.0);

    cv::Mat img_out;
    cv::Canny(img_in, img_out, threshold1, threshold2, apertureSize, L2gradient);

    img_out.convertTo(dataMat, CV_32F, 1.0 / 255.0);
    return 0;
}


int eogmaneo::OpenCVInterop::Threshold(
    std::vector<float>& data,
    float threshold, float maxValue,
    int type)
{
    // data contains float values [0.0 .. 1.0]
    cv::Mat dataMat(data, false);

    cv::Mat img_in;
    if (type & CV_THRESH_OTSU)
        // Otsu adaptive thresholding required 8-bit data
        dataMat.convertTo(img_in, CV_8U, 255.0);
    else
        dataMat.copyTo(img_in);

    cv::Mat img_out;
    cv::threshold(img_in, img_out, threshold, maxValue, type);

    img_out.convertTo(dataMat, CV_32F, 1.0 / 255.0);
    return 0;
}

int eogmaneo::OpenCVInterop::AdaptiveThreshold(
    std::vector<float>& data,
    float maxValue, int adaptiveMethod, int thresholdType,
    int blockSize, float C)
{
    // data contains float values [0.0 .. 1.0]
    cv::Mat dataMat(data, false);

    cv::Mat img_in;
    dataMat.convertTo(img_in, CV_8U, 255.0);

    cv::Mat img_out(img_in);
    cv::adaptiveThreshold(img_in, img_out, maxValue, adaptiveMethod, thresholdType, blockSize, C);

    img_out.convertTo(dataMat, CV_32F, 1.0 / 255.0);
    return 0;
}

int eogmaneo::OpenCVInterop::GaborFilter(
    std::vector<float>& data,
    int kernelSize, float sigma, float theta, float lambd, float gamma, float psi)
{
    // sigma controls the standard deviation of the Gaussian function used in the Gabor filter
    //   (width of the Gaussian envelope used in the Gabor kernel)
    // theta controls the orientation of the normal to the parallel stripes of the Gabor function
    //   (theta=0 makes filter responsive to horizontal features only)
    // lambda controls the wavelength of the sinusoidal factor in the above equation
    // gamma controls the ellipticity of the gaussian(the spatial aspect ratio)
    //   (gamma=1 makes the gaussian envelope circular)
    // psi controls the phase offset
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernelSize, kernelSize), sigma, theta, lambd, gamma, psi);

    cv::Mat img_in(data, true);
    cv::Mat img_out(img_in);
    cv::filter2D(img_in, img_out, CV_32F, kernel);

    img_out.convertTo(img_in, CV_32F, 1.0 / 255.0);
    data.assign((float*)img_in.datastart, (float*)img_in.dataend);
    return 0;
}


static cv::Ptr<cv::LineSegmentDetector> _LineSegmentDetector;

void eogmaneo::OpenCVInterop::LineSegmentDetector(
    std::vector<float>& data, int width, int height, int chunkSize,
    std::vector<int>& rotationSDR, bool drawLines)
{
    if (_LineSegmentDetector == nullptr) {
        // LSD_REFINE_NONE = No refinement applied
        // LSD_REFINE_STD  = Standard refinement is applied. E.g. breaking arches into
        //                   smaller straighter line approximations.
        // LSD_REFINE_ADV  = Advanced refinement. Number of false alarms is calculated, lines are
        //                   refined through increase of precision, decrement in size, etc.
        //
        _LineSegmentDetector = cv::createLineSegmentDetector(cv::LSD_REFINE_NONE);
    }

    // data contains float values [0.0 .. 1.0]
    cv::Mat dataMat(cv::Size(width, height), CV_32F, data.data());// , width * sizeof(float));

    cv::Mat img_in;
    dataMat.convertTo(img_in, CV_8U, 255.0f);

    std::vector<cv::Vec4f> lines;
    _LineSegmentDetector->detect(img_in, lines);

    if (drawLines) {
        // Show found lines
        cv::Mat linesMat = img_in.clone();// (cv::Size(width, height), CV_32F, data.data(), width * sizeof(float));
        _LineSegmentDetector->drawSegments(linesMat, lines);

        // Reverse channel expansion made in drawSegments
        cv::Mat img_out;
        cv::cvtColor(linesMat, img_out, CV_BGR2GRAY);

        // Overwrite input image data
        cv::Mat outMat;
        img_out.convertTo(outMat, CV_32FC1, 1.0f / 255.0f);
        outMat = outMat.reshape(0, 1);

        memcpy(data.data(), outMat.ptr<float>(0), width * height * sizeof(float));
    }

    // Zero rotation SDR bits
    rotationSDR.assign(rotationSDR.size(), 0);

    if (lines.size() > 0) {
        std::vector<float> chunkResponses;
        chunkResponses.assign(rotationSDR.size(), -99999.0f);

        int numChunksInX = (int)(width / chunkSize);
        int numChunksInY = (int)(height / chunkSize);
        int bitsPerChunk = chunkSize * chunkSize;
        float lineStepSize = chunkSize * 0.666f;
        float minLineLength = 6;

        for(auto line : lines) {
            cv::Point2f start(line[0], line[1]);
            cv::Point2f end(line[2], line[3]);

            cv::Point2f delta = end - start;
            float magnitude = delta.dot(delta);

            if (magnitude < minLineLength)
                continue;

            float response = magnitude;

            delta = (lineStepSize * delta) / std::max<float>(0.0001f, magnitude);

            float angle = atan2(delta.y, delta.x);

            int steps = (int)(magnitude / lineStepSize);

            cv::Point2f p(start);

            for (int s = 0; s < steps; s++) {
                // Fill
                int cx = std::min<int>(numChunksInX - 1, std::max<int>(0, int(p.x / chunkSize)));
                int cy = std::min<int>(numChunksInY - 1, std::max<int>(0, int(p.y / chunkSize)));

                int chunkIndex = cx + cy * numChunksInX;

                if (response > chunkResponses[chunkIndex]) {
                    chunkResponses[chunkIndex] = response;

                    rotationSDR[chunkIndex] = int(angle / (M_PI * 2.0f) * (bitsPerChunk - 1)) % bitsPerChunk;

                    if (rotationSDR[chunkIndex] < 0)
                        rotationSDR[chunkIndex] += bitsPerChunk;
                }

                // Step
                p += delta;
            }
        }
    }
}


// FAST - Detects corners using the FAST algorithm [Rosten06].
// Ref: Rosten, Machine Learning for High - speed Corner Detection, 2006.
//
// void FAST(InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression = true)
// Parameters :
//    image – grayscale image where keypoints(corners) are detected.
//    keypoints – keypoints detected on the image.
//    threshold – threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.
//    nonmaxSuppression – if true, non - maximum suppression is applied to detected corners(keypoints).
//    type – one of the three neighborhoods as defined in the paper : FastFeatureDetector::TYPE_9_16, FastFeatureDetector::TYPE_7_12, FastFeatureDetector::TYPE_5_8

void eogmaneo::OpenCVInterop::FastFeatureDetector(
    std::vector<float>& data, int width, int height, int chunkSize,
    std::vector<int>& featuresSDR, bool drawKeypoints, 
    int threshold, int type, bool nonmaxSuppression)
{
    // data contains float values [0.0 .. 1.0]
    cv::Mat dataMat(cv::Size(width, height), CV_32F, data.data());

    cv::Mat img_in;
    dataMat.convertTo(img_in, CV_8U, 255.0f);

    cv::GaussianBlur(img_in, img_in, cv::Size(3, 3), 0.0, 0.0);

    std::vector<cv::KeyPoint> keypoints;

    cv::FAST(img_in, keypoints, threshold, nonmaxSuppression, type);

    if (drawKeypoints) {
        // Show found lines
        cv::Mat img_out = img_in.clone();
        cv::drawKeypoints(img_in, keypoints, img_out);

        // Reverse any channel expansion
        cv::cvtColor(img_out, img_out, CV_BGR2GRAY);

        // Overwrite input image data
        cv::Mat outMat;
        img_out.convertTo(outMat, CV_32FC1, 1.0f / 255.0f);
        outMat = outMat.reshape(0, 1);

        memcpy(data.data(), outMat.ptr<float>(0), width * height * sizeof(float));
    }

    // Zero feature SDR bits
    featuresSDR.assign(featuresSDR.size(), 0);

    // Assign keypoints to SDR
    if (keypoints.size() > 0) {
        int numChunksInX = (int)(width / chunkSize);
        int numChunksInY = (int)(height / chunkSize);
        int bitsPerChunk = chunkSize * chunkSize;

        for (cv::KeyPoint l : keypoints) {
            cv::Point2f p = l.pt;

            // Fill
            int cx = std::min<int>(numChunksInX - 1, std::max<int>(0, int(p.x / chunkSize)));
            int cy = std::min<int>(numChunksInY - 1, std::max<int>(0, int(p.y / chunkSize)));

            int chunkIndex = cx + cy * numChunksInX;

            featuresSDR[chunkIndex] = std::min<int>(bitsPerChunk - 1, featuresSDR[chunkIndex] + 1);
        }
    }

}