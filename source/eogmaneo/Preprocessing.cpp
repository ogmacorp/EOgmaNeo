// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Preprocessing.h"

#include <algorithm>

using namespace eogmaneo;

void WhitenerWorkItem::run(size_t threadIndex) {
	whiten(*_psrc, *_pdest, _width, _radius, _strength, _cx, _cy, _chunkSize);
}

void SobelXWorkItem::run(size_t threadIndex) {
    sobelX(*_psrc, *_pdest, _width, _cx, _cy, _chunkSize);
}

void SobelYWorkItem::run(size_t threadIndex) {
    sobelY(*_psrc, *_pdest, _width, _cx, _cy, _chunkSize);
}

void SobelCombineWorkItem::run(size_t threadIndex) {
    sobelCombine(*_psrcX, *_psrcY, *_pdest, _width, _clip, _cx, _cy, _chunkSize);
}

std::vector<float> eogmaneo::whiten(const std::vector<float> &src, int width, int radius, float strength, ComputeSystem &system, int chunkSize) {
    std::vector<float> dest(src.size(), 0.0f);
	
	int height = src.size() / width;

	int chunksInX = std::ceil(static_cast<float>(width) / chunkSize);
	int chunksInY = std::ceil(static_cast<float>(height) / chunkSize);
	
	for (int cx = 0; cx < chunksInX; cx++)
		for (int cy = 0; cy < chunksInY; cy++) {
			std::shared_ptr<WhitenerWorkItem> item = std::make_shared<WhitenerWorkItem>();

			item->_cx = cx;
			item->_cy = cy;
			item->_psrc = &src;
			item->_pdest = &dest;
			item->_width = width;
			item->_radius = radius;
			item->_strength = strength;
			item->_chunkSize = chunkSize;

			system._pool.addItem(item);
		}
		
	system._pool.wait();
	
	return dest;
}

void eogmaneo::whiten(const std::vector<float> &src, std::vector<float> &dest, int width, int radius, float strength, int cx, int cy, int chunkSize) {
    int height = src.size() / width;

    for (int sx = 0; sx < chunkSize; sx++)
        for (int sy = 0; sy < chunkSize; sy++) {
            int x = cx * chunkSize + sx;
            int y = cy * chunkSize + sy;

            if (x >= 0 && y >= 0 && x < width && y < height) {
                float current = src[x + y * width];

                float center = 0.0f;
                float count = 0.0f;

                for (int dx = -radius; dx <= radius; dx++)
                    for (int dy = -radius; dy <= radius; dy++) {
                        if (dx == 0 && dy == 0)
                            continue;

                        int ox = x + dx;
                        int oy = y + dy;

                        if (ox >= 0 && oy >= 0 && ox < width && oy < height) {
                            center += src[ox + oy * width];
                            count += 1.0f;
                        }
                    }

                center /= std::max(0.0001f, count);

                float centeredCurrent = current - center;

                float covariance = 0.0f;

                for (int dx = -radius; dx <= radius; dx++)
                    for (int dy = -radius; dy <= radius; dy++) {
                        if (dx == 0 && dy == 0)
                            continue;

                        int ox = x + dx;
                        int oy = y + dy;

                        if (ox >= 0 && oy >= 0 && ox < width && oy < height) {
                            float centeredOther = src[ox + oy * width] - center;

                            covariance += centeredCurrent * centeredOther;
                        }
                    }

                float whitened = std::min(1.0f, std::max(-1.0f, (centeredCurrent > 0.0f ? 1.0f : -1.0f) * (1.0f - std::exp(-strength * std::abs(covariance)))));
            
                dest[x + y * width] = whitened;
            }
        }
}

std::vector<float> eogmaneo::sobel(const std::vector<float> &src, int width, float clip, ComputeSystem &system, int chunkSize) {
    std::vector<float> destX(src.size(), 0.0f);
    std::vector<float> destY(src.size(), 0.0f);
    std::vector<float> dest(src.size(), 0.0f);

    int height = src.size() / width;

    int chunksInX = std::ceil(static_cast<float>(width) / chunkSize);
    int chunksInY = std::ceil(static_cast<float>(height) / chunkSize);

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            // X
            {
                std::shared_ptr<SobelXWorkItem> item = std::make_shared<SobelXWorkItem>();

                item->_cx = cx;
                item->_cy = cy;
                item->_psrc = &src;
                item->_pdest = &destX;
                item->_width = width;
                item->_chunkSize = chunkSize;

                system._pool.addItem(item);
            }

            // Y
            {
                std::shared_ptr<SobelYWorkItem> item = std::make_shared<SobelYWorkItem>();

                item->_cx = cx;
                item->_cy = cy;
                item->_psrc = &src;
                item->_pdest = &destY;
                item->_width = width;
                item->_chunkSize = chunkSize;

                system._pool.addItem(item);
            }
        }

    system._pool.wait();

    for (int cx = 0; cx < chunksInX; cx++)
        for (int cy = 0; cy < chunksInY; cy++) {
            std::shared_ptr<SobelCombineWorkItem> item = std::make_shared<SobelCombineWorkItem>();

            item->_cx = cx;
            item->_cy = cy;
            item->_psrcX = &destX;
            item->_psrcY = &destY;
            item->_pdest = &dest;
            item->_width = width;
            item->_clip = clip;
            item->_chunkSize = chunkSize;

            system._pool.addItem(item);
        }

    system._pool.wait();

    return dest;
}

void eogmaneo::sobelX(const std::vector<float> &src, std::vector<float> &dest, int width, int cx, int cy, int chunkSize) {
    int height = src.size() / width;

    std::vector<std::vector<float>> kernel = {
        { 1.0f, 0.0f, -1.0f },
        { 2.0f, 0.0f, -2.0f },
        { 1.0f, 0.0f, -1.0f }
    };

    for (int sx = 0; sx < chunkSize; sx++)
        for (int sy = 0; sy < chunkSize; sy++) {
            int x = cx * chunkSize + sx;
            int y = cy * chunkSize + sy;

            if (x >= 0 && y >= 0 && x < width && y < height) {
                float output = 0.0f;

                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++) {
                        int ox = x + dx;
                        int oy = y + dy;

                        float scale = kernel[dx + 1][dy + 1];

                        if (ox >= 0 && oy >= 0 && ox < width && oy < height)
                            output += scale * src[ox + oy * width];
                    }

                dest[x + y * width] = output;
            }
        }
}

void eogmaneo::sobelY(const std::vector<float> &src, std::vector<float> &dest, int width, int cx, int cy, int chunkSize) {
    int height = src.size() / width;

    std::vector<std::vector<float>> kernel = {
        { 1.0f, 2.0f, 1.0f },
        { 0.0f, 0.0f, 0.0f },
        { -1.0f, -2.0f, -1.0f }
    };

    for (int sx = 0; sx < chunkSize; sx++)
        for (int sy = 0; sy < chunkSize; sy++) {
            int x = cx * chunkSize + sx;
            int y = cy * chunkSize + sy;

            if (x >= 0 && y >= 0 && x < width && y < height) {
                float output = 0.0f;

                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++) {
                        int ox = x + dx;
                        int oy = y + dy;

                        float scale = kernel[dx + 1][dy + 1];

                        if (ox >= 0 && oy >= 0 && ox < width && oy < height)
                            output += scale * src[ox + oy * width];
                    }

                dest[x + y * width] = output;
            }
        }
}

void eogmaneo::sobelCombine(const std::vector<float> &srcX, const std::vector<float> &srcY, std::vector<float> &dest, int width, float clip, int cx, int cy, int chunkSize) {
    int height = srcX.size() / width;

    for (int sx = 0; sx < chunkSize; sx++)
        for (int sy = 0; sy < chunkSize; sy++) {
            int x = cx * chunkSize + sx;
            int y = cy * chunkSize + sy;

            if (x >= 0 && y >= 0 && x < width && y < height) {
                float output = std::sqrt(srcX[x + y * width] * srcX[x + y * width] + srcY[x + y * width] * srcY[x + y * width]);

                dest[x + y * width] = std::max(0.0f, output * (1.0f + clip) - clip);
            }
        }
}