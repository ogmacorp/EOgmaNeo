// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "ThreadPool.h"

#include <random>

namespace eogmaneo {
    class ThreadPool;

	/*!
	\brief Compute system. Mainly passed to other functions. Contains thread pooling and random number generator information.
	*/
    class ComputeSystem {
	private:
		ThreadPool _pool;
		std::mt19937 _rng;

	public:
		/*!
		\brief Initialize the system.
		\param numWorkers number of thread pool worker threads.
		\param seed global random number generator seed. Defaults to 1234.
		*/
        ComputeSystem(size_t numWorkers, unsigned long seed = 1234) {
			_pool.create(numWorkers);
			_rng.seed(seed);
		}
		
		friend class Layer;
		friend class Hierarchy;
		
		friend class KMeansEncoder;
		friend class ImageEncoder;
		friend class GaborEncoder;
    };
}