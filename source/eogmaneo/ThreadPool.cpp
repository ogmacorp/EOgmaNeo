// ----------------------------------------------------------------------------
//  EOgmaNeo
//  Copyright(c) 2017-2018 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of EOgmaNeo is licensed to you under the terms described
//  in the EOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "ThreadPool.h"

#include <iostream>

using namespace eogmaneo;

void WorkerThread::run(WorkerThread* pWorker) {
	while (true) {
		std::unique_lock<std::mutex> lock(pWorker->_mutex);

		pWorker->_conditionVariable.wait(lock, [pWorker] { return static_cast<bool>(pWorker->_proceed); });

		pWorker->_proceed = false;

		if (pWorker->_pPool == nullptr)
			break;
		else {
			if (pWorker->_item != nullptr) {
				pWorker->_item->run(pWorker->_workerIndex);
				pWorker->_item->_done = true;
			}

			pWorker->_pPool->onWorkerAvailable(pWorker->_workerIndex);
		}

		pWorker->_conditionVariable.notify_one();
	}
}

void ThreadPool::onWorkerAvailable(size_t workerIndex) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (_itemQueue.empty())
		_availableThreadIndicies.push_back(workerIndex);
	else {
		// Assign new task
		_workers[workerIndex]->_item = _itemQueue.front();
		_itemQueue.pop_front();
		_workers[workerIndex]->_proceed = true;
	}
}

void ThreadPool::create(size_t numWorkers) {
	_workers.resize(numWorkers);

	// Add all threads as available and launch threads
	for (size_t i = 0; i < _workers.size(); i++) {
		_workers[i].reset(new WorkerThread());

		_availableThreadIndicies.push_back(i);

		// Block all threads as there are no tasks yet
		_workers[i]->_pPool = this;
		_workers[i]->_workerIndex = i;

		_workers[i]->start();
	}
}

void ThreadPool::destroy() {
	//std::lock_guard<std::mutex> lock(_mutex);

	_itemQueue.clear();
	_availableThreadIndicies.clear();

	for (size_t i = 0; i < _workers.size(); i++) {
		{
			std::lock_guard<std::mutex> lock(_workers[i]->_mutex);
			_workers[i]->_item = nullptr;
			_workers[i]->_pPool = nullptr;
			_workers[i]->_proceed = true;
			_workers[i]->_conditionVariable.notify_one();
		}

		_workers[i]->_thread->join();
	}
}

void ThreadPool::addItem(const std::shared_ptr<WorkItem> &item) {
	std::lock_guard<std::mutex> lock(_mutex);

	if (workersAvailable()) {
		size_t workerIndex = _availableThreadIndicies.front();

		_availableThreadIndicies.pop_front();

		std::lock_guard<std::mutex> lock(_workers[workerIndex]->_mutex);

		_workers[workerIndex]->_item = item;
		_workers[workerIndex]->_proceed = true;
		_workers[workerIndex]->_conditionVariable.notify_one();
	}
	else
		_itemQueue.push_back(item);
}

void ThreadPool::wait() {
	// Try to aquire every mutex until no tasks are left
	while (true) {
		for (size_t i = 0; i < _workers.size(); i++) {
			std::unique_lock<std::mutex> lock(_workers[i]->_mutex);

			WorkerThread* pWorker = _workers[i].get();

			_workers[i]->_conditionVariable.wait(lock, [pWorker] { return !pWorker->_proceed; });
		}

		std::lock_guard<std::mutex> lock(_mutex);

		if (_itemQueue.empty())
			break;
	}
}