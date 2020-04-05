/*
 * atomicLock.h
 *
 *  Created on: Mar 26, 2020
 *      Author: yzchen
 */

#ifndef ATOMICLOCK_H_
#define ATOMICLOCK_H_
#include <cuda_runtime.h>
#include <assert.h>

typedef int Lock;

template <int N>
struct LockSet {
  enum {
    CLS_FREE = 0,
    CLS_LOCKED = -1,
  };
	Lock *locks[N];

	__host__ __device__
	LockSet()
 	: locks() {
	}

	__host__ __device__
	~LockSet() {
		YieldAll();
	}

	/**
   * add a lock to the list
	 * */
	__device__ __host__
	bool TryLock(Lock &lock) {
#ifdef __CUDA_ARCH__
		int final_pos = -1;

		// insert into table
		for (int i=0; i<N; i++) {
			if (locks[i] == 0) {
				locks[i] = &lock;
				final_pos = i;
				break;
			}
		}

		// no space
    assert(final_pos != -1);

		// check if already acquired (write)
		for (int i=0; i<N; i++) {
			if (locks[i] == &lock && i != final_pos) {
        return true;
			}
		}
		//atomicCAS(a, b, c)将判断变量a是否等于b，
		 //若相等，则用c的值去替换a，并返回a的值；若不相等，则返回a的值。 所以总是return a的旧值。
		 //函数lock()中，线程不断尝试判断mutex是否为0，
		 //若为0则改写为-1 ，表明“占用”，禁止其他线程进行访问
		 //若为-1则继续尝试判断
		//old == compare ? val : old
    // if not acquired, then we should switch it to LOCKED
		// try to write
		// the attempt invalidates ALL read locks -- because a writer is already going to change the values

		int result = atomicCAS(&lock, (int)CLS_FREE, (int)CLS_LOCKED);

		if (result == CLS_LOCKED) { // failed to acquire
			// invalidate this lock
			locks[final_pos] = 0;
      return false;
		}
    else if (result == CLS_FREE) { // acquired
			return true;
		}
    else {
      assert(false);
      return false;
    }
#else
    // FIXME: implement c++11 <atomic> or something
		return true;
#endif
	}

	/**
	 * Yield the specified lock
	 *
	 * First releases the write locks
	 * then releases the read locks
	 *
	 * */
	__device__ __host__
	void Yield(Lock &lock) {
#ifdef __CUDA_ARCH__
		// check if already acquired (write)
		for (int i=0; i<N; i++) {
			if (locks[i] == &lock) {

				// check that we are not double-freeing
				for (int j=i+1; j<N; j++) {
					if (locks[j] == &lock) {
						locks[j] = 0;
						return;
					}
				}

				// no 2nd lock -- free this lock
				_YieldLock(lock);
				locks[i] = 0;
        return;
			}
		}
#endif
	}

#ifdef __CUDACC__
	__device__
	void _YieldLock(Lock &lock) {
		//atomicExch(a, b)返回第一个变量的值，并将两个变量的值进行交换
		 //这里使用原子操作只是与上面的atomicCAS统一，否则可以直接用赋值语句
		 //线程操作完成，将mutex改写回0，允许其他线程进行访问
    int result = atomicExch(&lock, (int)CLS_FREE);
    assert(result == (int)CLS_LOCKED);
	}
#endif

	/**
	 * */
	__device__ __host__
	void YieldAll() {
#ifdef __CUDA_ARCH__
		// yield writers
		for (int i=0; i<N; i++) {
			if (locks[i]) {
				_YieldLock(*locks[i]);

				for (int j=i+1; j<N; j++) { // avoid double-free
					if (locks[i] == locks[j]) {
						locks[j] = 0; // clear additional write locks
					}
				}
				locks[i] = 0;
			}
		}
#endif
	}
};






#endif /* ATOMICLOCK_H_ */
