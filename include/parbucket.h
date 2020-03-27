#ifndef PARBUCKET_H
#define PARBUCKET_H


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <utils.h>
#include <math.h>
#include <cuda_utils.h>
#include <MinHeap.h>
#include <iostream>
#include "atomicLock.h"

template <class Ktype>
struct __align__(16) VoxBucketItem {
	Ktype key;
	int priority;
};

//template <class Ktype>
//struct __align__(32) VoxSignalItem
//{
//    int signalType;
//    Ktype key;
//    int priority;
//    bool handled=false;
// public:
//    __device__ __host__
//    VoxSignalItem(int sigType, Ktype key_, int priority_):
//        signalType(sigType),priority(priority_),key(key_){}

//    int getSignalType() const { return signalType; }
//    Ktype getKey() { return key; }
//    int getPriority() const { return priority; }
//    bool getHandled() { return handled; }

//    void setHandled(bool handled_) {handled = handled_;}
//    void setSignalType(int sigType){signalType = sigType;}

//};

//template <class Ktype>
//struct SignalLine
//{
//    VoxSignalItem<Ktype>* voxSignals;
//    int capacity;
//    int active_num;
//public:
//    SignalLine(int capacity_):capacity(capacity_)
//    {
//        int mem_size=sizeof(VoxSignalItem<Ktype>)*capacity;
//        CUDA_ALLOC_DEV_MEM(voxSignals,mem_size);
//    }
//    ~SignalLine()
//    {
//        CUDA_FREE_DEV_MEM(voxSignals);
//    }
//    void insertKey(int signalType, Ktype key, int priority);
//    void reset();
//    void deleteHandledSignals();

//    bool isEmpty(){ return active_num == 0;}
//    int getSize() {return active_num;}
//    int getCapacity() {return capacity;}
//    VoxSignalItem<Ktype> getSignal(int index){ return voxSignals[index];}
//    void setHandled(int index, bool handled){voxSignals[index].setHandled(handled);}
//    void setSignalType(int index, int signalType){voxSignals[index].setSignalType(signalType);}

//};
//template <class Ktype>
//struct BucketLine
//{
//    VoxBucketItem<Ktype> *voxBuckets;// heap array
//    int heap_size; // Current number of elements in BucketLine
//    int active_num;
//public:
//    BucketLine(int capacity) :heap_size(capacity),active_num(0)
//    {
//        int mem_size=sizeof(VoxBucketItem<Ktype>)*heap_size;
//        CUDA_ALLOC_DEV_MEM(voxBuckets,mem_size);

//    }
//    ~BucketLine()
//    {
//        CUDA_FREE_DEV_MEM(voxBuckets);
//    }
//    __device__
//    VoxBucketItem<Ktype> getItem(int index) {return voxBuckets[index];}
//    int getMaxPriority();
//    int getIthSmallestPriority(int i);

//    int moveGreaterPriorityTo(SignalLine<Ktype>* SiPlus1, int priority);
//    int moveSimilarPriorityTo(SignalLine<Ktype>* SiPlus1, int priority, int itemsToMove);

//    int moveSmallerPriorityFromBucket(BucketLine *BiPlus1, int priority, int i);
//    int moveSimilarPriorityFromBucket(BucketLine *BiPlus1, int priority, int i);
//};

template <class Ktype>
struct BSlevel
{
	VoxBucketItem<Ktype> *voxBuckets;
	VoxBucketItem<Ktype> *voxSignals;
	//    thrust::device_vector<VoxBucketItem<Ktype>> vecBuckets;
	//    thrust::device_vector<VoxBucketItem<Ktype>> vecSignals;

	int bucketIndex;
	int bucketCapacity,active_buckets;
	int signalCapacity,active_signals;
	int timestamp;

	int *d_bucketIndex;
	int *d_bucketCapacity,*d_active_buckets;
	int *d_signalCapacity,*d_active_signals;
	int *d_timestamp;

	int *test_h,*test_d;
	BSlevel(int id,int d):bucketIndex(id),timestamp(0)
	{
		bucketCapacity = d*pow(2, (2*id+1));     // Bi
		signalCapacity = d*pow(2, (2*id)); // Si+1
		active_buckets=0;
		active_signals=0;
		//        vecBuckets.resize(bucketCapacity);
		//        vecSignals.resize(signalCapacity);
		int bucket_size=sizeof(VoxBucketItem<Ktype>)*bucketCapacity;
		CUDA_ALLOC_DEV_MEM(&voxBuckets,bucket_size);
		int siganl_size=sizeof(VoxBucketItem<Ktype>)*signalCapacity;
		CUDA_ALLOC_DEV_MEM(&voxSignals,siganl_size);


		param_initD();
		param_H2D();
//		IncStamp();

		test_h=new int[1];
		test_h[0]=38;
		CUDA_ALLOC_DEV_MEM(&test_d,sizeof(int));
		CUDA_MEMCPY_H2D(test_d,test_h,sizeof(int));

	}
	~BSlevel()
	{
		CUDA_FREE_DEV_MEM(voxBuckets);
		CUDA_FREE_DEV_MEM(voxSignals);
		param_deinitD();
	}
	__host__
	void param_initD()
	{
		CUDA_ALLOC_DEV_MEM(&d_bucketIndex,sizeof(int));
		CUDA_ALLOC_DEV_MEM(&d_bucketCapacity,sizeof(int));
		CUDA_ALLOC_DEV_MEM(&d_active_buckets,sizeof(int));
		CUDA_ALLOC_DEV_MEM(&d_signalCapacity,sizeof(int));
		CUDA_ALLOC_DEV_MEM(&d_active_signals,sizeof(int));
		CUDA_ALLOC_DEV_MEM(&d_timestamp,sizeof(int));
	}
	__host__
	void param_H2D()
	{
		CUDA_MEMCPY_H2D(d_bucketIndex,&bucketIndex,sizeof(int));
		CUDA_MEMCPY_H2D(d_bucketCapacity,&bucketCapacity,sizeof(int));
		CUDA_MEMCPY_H2D(d_active_buckets,&active_buckets,sizeof(int));
		CUDA_MEMCPY_H2D(d_signalCapacity,&signalCapacity,sizeof(int));
		CUDA_MEMCPY_H2D(d_active_signals,&active_signals,sizeof(int));
		CUDA_MEMCPY_H2D(d_timestamp,&timestamp,sizeof(int));
	}
	__host__
	void param_D2H()
	{
		CUDA_MEMCPY_D2H(&bucketIndex,d_bucketIndex,sizeof(int));
		CUDA_MEMCPY_D2H(&bucketCapacity,d_bucketCapacity,sizeof(int));
		CUDA_MEMCPY_D2H(&active_buckets,d_active_buckets,sizeof(int));
		CUDA_MEMCPY_D2H(&signalCapacity,d_signalCapacity,sizeof(int));
		CUDA_MEMCPY_D2H(&active_signals,d_active_signals,sizeof(int));
		CUDA_MEMCPY_D2H(&timestamp,d_timestamp,sizeof(int));
	}
	__host__
	void param_deinitD()
	{
		CUDA_FREE_DEV_MEM(d_bucketIndex);
		CUDA_FREE_DEV_MEM(d_bucketCapacity);
		CUDA_FREE_DEV_MEM(d_active_buckets);
		CUDA_FREE_DEV_MEM(d_signalCapacity);
		CUDA_FREE_DEV_MEM(d_active_signals);
		CUDA_FREE_DEV_MEM(d_timestamp);
		CUDA_FREE_DEV_MEM(test_d);
	}
	__device__
	void insertKey(Ktype k, int p)
	{

	}
	__device__
	void Merge(bool output_Bi,VoxBucketItem<Ktype>* harr1,VoxBucketItem<Ktype>* harr2)
	{

	}
	__device__
	void clearS()
	{

	}
	__device__
	void DelDupB()
	{

	}
	__device__
	void IncStamp()
	{
		d_timestamp[0]++;
	}
	__device__ int getTimeStamp()
	{
		int ts=d_timestamp[0];
		return ts;
	}


};
template <class Ktype>
struct ParBucketHeap
{
	//    int num_buckets;
	int q; // largest non-empty level of Bi
	int max_level;// maximum level size
	int d; //bulk size

	int *d_q; // largest non-empty level of Bi
	int *d_max_level;// maximum level size
	int *d_d; //bulk size


	int* d_priorities;
	int* d_locks;
	int* d_mem;

	BSlevel<Ktype>* bs_levels;
	ParBucketHeap(int n,int d_=1):d(d_),q(-1)
	{

		max_level=log(n/d)/log(4)+1;
		for(int i=0;i<max_level;i++)
		{
			new (&bs_levels[i]) BSlevel<Ktype>(i,d);
		}
		CUDA_ALLOC_DEV_MEM(&d_priorities,sizeof(int)*max_level);
		int cuncurrent_levels=ceil(1.0f*max_level/2);
		CUDA_ALLOC_DEV_MEM(&d_locks,sizeof(int)*cuncurrent_levels);
		CUDA_ALLOC_DEV_MEM(&d_mem,sizeof(int)*cuncurrent_levels);
		CUDA_DEV_MEMSET(d_mem,0,sizeof(int)*cuncurrent_levels);
		param_initD();
		param_H2D();

	}
	~ParBucketHeap()
	{
		delete [] bs_levels;
		CUDA_FREE_DEV_MEM(d_priorities);
		CUDA_FREE_DEV_MEM(d_locks);
		CUDA_FREE_DEV_MEM(d_mem);
		param_deinitD();
	}
	__host__
	void param_initD()
	{
		CUDA_ALLOC_DEV_MEM(&d_q,sizeof(int));
		CUDA_ALLOC_DEV_MEM(&d_max_level,sizeof(int));
		CUDA_ALLOC_DEV_MEM(&d_d,sizeof(int));
	}
	__host__
	void param_H2D()
	{
		CUDA_MEMCPY_H2D(d_q,&q,sizeof(int));
		CUDA_MEMCPY_H2D(d_max_level,&max_level,sizeof(int));
		CUDA_MEMCPY_H2D(d_d,&d,sizeof(int));
	}
	__host__
	void param_D2H()
	{
		CUDA_MEMCPY_D2H(&q,d_q,sizeof(int));
		CUDA_MEMCPY_D2H(&max_level,d_max_level,sizeof(int));
		CUDA_MEMCPY_D2H(&d,d_d,sizeof(int));
	}
	__host__
	void param_deinitD()
	{
		CUDA_FREE_DEV_MEM(d_q);
		CUDA_FREE_DEV_MEM(d_max_level);
		CUDA_FREE_DEV_MEM(d_d);
	}
	__device__
	bool metConstrain(int level)
	{
		int this_cnt,last_cnt,next_cnt;
		this_cnt=bs_levels[level].getTimeStamp();
		// 1st level
		if(level==0)
		{
			next_cnt=bs_levels[level+1].d_timestamp[0];
			if(next_cnt>=4*this_cnt)
				return true;
			else
				return false;
		}
		// last level
		if(level==d_q[0]-1)
		{
			last_cnt=bs_levels[level-1].d_timestamp[0];
			if(last_cnt==4*this_cnt)
				return true;
			else
				return false;
		}//else
			next_cnt=bs_levels[level+1].d_timestamp[0];
			last_cnt=bs_levels[level-1].d_timestamp[0];
			if(last_cnt==4*this_cnt&&next_cnt>=4*this_cnt)
				return true;
			else
				return false;
	}
	__device__
	void bulkUpdate(Ktype k, int p)
	{
		bs_levels->insertKey(k,p);
	}
	Ktype deleteMin();
	void deleteItem(Ktype k);

	void empty(int signalIndex);
	void fill(int bucketIndex);
	__device__
	int smallerP(int level)
	{
		return 0;
	}
	__device__
	int SelectP(int level)
	{
		return 0;
	}
	__device__
	VoxBucketItem<Ktype>* SeprateB(int this_level,int p_i)
	{
		VoxBucketItem<Ktype>* Bi_dot;
		return Bi_dot;
	}
	__device__
	int largest_active_level()
	{
		return 0;
	}
	__device__
	int Resolve(int level)
	{
		if(!metConstrain(level))
			return -1;


		/////////////////////// test
		// if locked, wait
		LockSet<1> lockset;
		while(!lockset.TryLock(d_locks[level/2]))
		{
			// waiting for adjacent blk
		}

		for(int i=0;i<100;i++)
		{
			// do something on mem
			d_mem[level/2]++;
		}
		int out=d_mem[level/2];
		d_mem[level/2]=0;

		bs_levels[level].IncStamp();
		//unlock
		lockset.Yield(d_locks[level/2]);


		return out;
		///////////////////////
		// empty if needed
		if(bs_levels[level].d_active_signals[0]==0)
		{
			VoxBucketItem<Ktype>* Si=bs_levels[level].voxSignals;
			VoxBucketItem<Ktype>* Bi=bs_levels[level].voxBuckets;
			bs_levels[level].Merge(true,Bi,Si);
			bs_levels[level].DelDupB();

			int num=smallerP(level);
			if(num>pow(2,2*level+1))
			{
				d_priorities[level]=SelectP(level);
				VoxBucketItem<Ktype>* Bi_dot=SeprateB(level,d_priorities[level]);
				VoxBucketItem<Ktype>* Siplus1=bs_levels[level+1].voxSignals;
				bs_levels[level+1].Merge(false,Siplus1,Bi_dot);
			}


		}

		// fill if needed
		int BiSize=bs_levels[level].d_active_buckets[0];
		d_q[0]=largest_active_level();
		if(BiSize<pow(2,2*level+1)&&level<d_q[0]-1)// Bi not enough, level is not largest non_empty
		{

		}
		return 0;
	}
	//    BucketLine<Ktype>* getIthBucket(int i);
	//    SignalLine<Ktype>* getIthSignal(int i);

	int getMaxPriorityOnBucket(int i, int q);

	int getNonEmptyBucketSignalIndex();

	void maintainNumBuckets();
};

int parDijkstra();
#endif // PARBUCKET_H
