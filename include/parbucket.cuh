#pragma once

#include <utils.h>
#include <cuda_utils.h>
#include <MinHeap.h>
#include "atomicLock.h"



template <class Ktype>
struct __align__(16) VoxBucketItem {
	Ktype key;
	int priority;
//	__device__ __host__
//	VoxBucketItem(Ktype k,int p):key(k),priority(p)
//	{
//
//	}
	__device__ __host__
	void setVal(Ktype k,int p)
	{
		key=k;
		priority=p;
	}
};



//template <class Ktype>
//struct BSlevel
//{
//	VoxBucketItem<Ktype> *voxBuckets;
//	VoxBucketItem<Ktype> *voxSignals;
//	//    thrust::device_vector<VoxBucketItem<Ktype>> vecBuckets;
//	//    thrust::device_vector<VoxBucketItem<Ktype>> vecSignals;
//
//	int bucketIndex;
//	int bucketCapacity,active_buckets;
//	int signalCapacity,active_signals;
//	int timestamp;
//
//	int *d_bucketIndex;
//	int *d_bucketCapacity,*d_active_buckets;
//	int *d_signalCapacity,*d_active_signals;
//	int *d_timestamp;
//
//
//	BSlevel(int id,int d):bucketIndex(id),timestamp(0)
//	{
//		bucketCapacity = d*pow(2, (2*id+1));     // Bi
//		signalCapacity = d*pow(2, (2*id)); // Si+1
//		active_buckets=0;
//		active_signals=0;
//		//        vecBuckets.resize(bucketCapacity);
//		//        vecSignals.resize(signalCapacity);
//		int bucket_size=sizeof(VoxBucketItem<Ktype>)*bucketCapacity;
//		CUDA_ALLOC_DEV_MEM(&voxBuckets,bucket_size);
//		int siganl_size=sizeof(VoxBucketItem<Ktype>)*signalCapacity;
//		CUDA_ALLOC_DEV_MEM(&voxSignals,siganl_size);
//
//
//		param_initD();
//		param_H2D();
////		IncStamp();
//
//	}
//	~BSlevel()
//	{
//		CUDA_FREE_DEV_MEM(voxBuckets);
//		CUDA_FREE_DEV_MEM(voxSignals);
//		param_deinitD();
//	}
//	__host__
//	void param_initD()
//	{
//		CUDA_ALLOC_DEV_MEM(&d_bucketIndex,sizeof(int));
//		CUDA_ALLOC_DEV_MEM(&d_bucketCapacity,sizeof(int));
//		CUDA_ALLOC_DEV_MEM(&d_active_buckets,sizeof(int));
//		CUDA_ALLOC_DEV_MEM(&d_signalCapacity,sizeof(int));
//		CUDA_ALLOC_DEV_MEM(&d_active_signals,sizeof(int));
//		CUDA_ALLOC_DEV_MEM(&d_timestamp,sizeof(int));
//	}
//	__host__
//	void param_H2D()
//	{
//		CUDA_MEMCPY_H2D(d_bucketIndex,&bucketIndex,sizeof(int));
//		CUDA_MEMCPY_H2D(d_bucketCapacity,&bucketCapacity,sizeof(int));
//		CUDA_MEMCPY_H2D(d_active_buckets,&active_buckets,sizeof(int));
//		CUDA_MEMCPY_H2D(d_signalCapacity,&signalCapacity,sizeof(int));
//		CUDA_MEMCPY_H2D(d_active_signals,&active_signals,sizeof(int));
//		CUDA_MEMCPY_H2D(d_timestamp,&timestamp,sizeof(int));
//	}
//	__host__
//	void param_D2H()
//	{
//		CUDA_MEMCPY_D2H(&bucketIndex,d_bucketIndex,sizeof(int));
//		CUDA_MEMCPY_D2H(&bucketCapacity,d_bucketCapacity,sizeof(int));
//		CUDA_MEMCPY_D2H(&active_buckets,d_active_buckets,sizeof(int));
//		CUDA_MEMCPY_D2H(&signalCapacity,d_signalCapacity,sizeof(int));
//		CUDA_MEMCPY_D2H(&active_signals,d_active_signals,sizeof(int));
//		CUDA_MEMCPY_D2H(&timestamp,d_timestamp,sizeof(int));
//	}
//	__host__
//	void param_deinitD()
//	{
//		CUDA_FREE_DEV_MEM(d_bucketIndex);
//		CUDA_FREE_DEV_MEM(d_bucketCapacity);
//		CUDA_FREE_DEV_MEM(d_active_buckets);
//		CUDA_FREE_DEV_MEM(d_signalCapacity);
//		CUDA_FREE_DEV_MEM(d_active_signals);
//		CUDA_FREE_DEV_MEM(d_timestamp);
//	}
//	__device__
//	void insertKey(Ktype k, int p)
//	{
//
//	}
//	__device__
//	void Merge(bool output_Bi,VoxBucketItem<Ktype>* harr1,VoxBucketItem<Ktype>* harr2)
//	{
//
//	}
//	__device__
//	void clearS()
//	{
//
//	}
//	__device__
//	void DelDupB()
//	{
//
//	}

//
//
//};


template <class Ktype>
struct ParBucketHeapBase
{

	// param for the whole heap
	int q; // largest non-empty level of Bi
	int d; //bulk size

	// param for each level
	// length==levels


	int *d_bucketOffsets;
	int *d_signalOffsets;
	int* d_priorities;
	int *d_active_buckets;
	int *d_active_signals;
	int *d_timestamps;

	// mutex,length==levels/2
	int* d_locks;
	int* d_mem;


	// buckets and signals
	VoxBucketItem<Ktype> *voxBuckets;
	VoxBucketItem<Ktype> *voxSignals;

	//	BSlevel<Ktype>* bs_levels;
	ParBucketHeapBase(int n,int d_=1):d(d_),q(-1)
	{

	}

	__device__ VoxBucketItem<Ktype> *getBucItem(int lv,int id)
		{
			int glbId=d_bucketOffsets[lv]+id;
			return &(voxBuckets[glbId]);
		}

	__device__ VoxBucketItem<Ktype> *getSigItem(int lv,int id)
		{
			int glbId=d_signalOffsets[lv]+id;
			return &(voxSignals[glbId]);
		}

	__device__ int update(VoxBucketItem<Ktype> *eInPtr)
	{
		if(d_active_signals[0]>0)
		{
			printf("update precondition not met!");
			assert(false);
			return -1;
		}
		VoxBucketItem<Ktype>* S0=getSigItem(0,0);
		S0->setVal(eInPtr->key,eInPtr->priority);
		d_active_signals[0]++;
		return 0;
	}
	__device__ int  findMin()
		{
		// TODO reduce
		int sz=2*d;
		int minPriority=INT_MAX;
		int idOut=0;
		for(int i=0;i<sz;i++)
		{
			VoxBucketItem<Ktype>* elem=getBucItem(0,i);
			if(elem->priority<minPriority)
			{
				minPriority=elem->priority;
				idOut=i;
			}
		}
		return idOut;

		}
	__device__ int  extractMin(VoxBucketItem<Ktype> &eOut)
	{
		if(d_active_buckets[0]<d || q<=0  )
		{
			printf("extractMin precondition 1 not met!");
			return -1;
		}
		if(d_active_signals[0]!=0)
		{
			printf("extractMin precondition 2 not met!");
			return -2;
		}

//		VoxBucketItem<Ktype> eOut;
		int idOut=findMin();
		VoxBucketItem<Ktype>* eOutPtr=getBucItem(0,idOut);

		VoxBucketItem<Ktype>* S0=getSigItem(0,0);
		S0->setVal(eOutPtr->key,-1 );
		d_active_signals[0]++;

		eOut.setVal(eOutPtr->key,eOutPtr->priority);
		removeElemB(0,idOut);
		return 0;
	}

	__device__ void removeElemB(int lv,int id)
	{
		VoxBucketItem<Ktype>* toRemove=getBucItem(lv,id);
		toRemove->key=0;
		toRemove->priority=0;
		d_active_buckets--;
	}
	//delete not implemented yet

	__device__ int getTimeStamp(int level)
	{
		return d_timestamps[level];
	}
	__device__
	bool metConstrain(int level)
	{
		int this_cnt,last_cnt,next_cnt;
		this_cnt=getTimeStamp(level);
		// 1st level
		if(level==0)
		{
			next_cnt=getTimeStamp(level+1);
			if(next_cnt>=4*this_cnt)
				return true;
			else
				return false;
		}
		// last level
		if(level==q-1)
		{
			last_cnt=getTimeStamp(level-1);
			if(last_cnt==4*this_cnt)
				return true;
			else
				return false;
		}//else
		next_cnt=getTimeStamp(level+1);
		last_cnt=getTimeStamp(level-1);
		if(last_cnt==4*this_cnt&&next_cnt>=4*this_cnt)
			return true;
		else
			return false;
	}

	__device__ void IncStamp(int lv)
	{
		d_timestamps[lv]++;
	}
	__device__
	void assertNeetRes(int lv)
	{
		bool wait=1;
		do
		{

		}while(wait);

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
		int out=d_mem[level/2]*getTimeStamp(level);
		d_mem[level/2]=0;

		IncStamp(level);
		//unlock
		lockset.Yield(d_locks[level/2]);


		return out;
		///////////////////////
		// empty if needed
		//		if(bs_levels[level].d_active_signals[0]==0)
		//		{
		//			VoxBucketItem<Ktype>* Si=bs_levels[level].voxSignals;
		//			VoxBucketItem<Ktype>* Bi=bs_levels[level].voxBuckets;
		//			bs_levels[level].Merge(true,Bi,Si);
		//			bs_levels[level].DelDupB();
		//
		//			int num=smallerP(level);
		//			if(num>pow(2,2*level+1))
		//			{
		//				d_priorities[level]=SelectP(level);
		//				VoxBucketItem<Ktype>* Bi_dot=SeprateB(level,d_priorities[level]);
		//				VoxBucketItem<Ktype>* Siplus1=bs_levels[level+1].voxSignals;
		//				bs_levels[level+1].Merge(false,Siplus1,Bi_dot);
		//			}
		//
		//
		//		}
		//
		//		// fill if needed
		//		int BiSize=bs_levels[level].d_active_buckets[0];
		//		d_q[0]=largest_active_level();
		//		if(BiSize<pow(2,2*level+1)&&level<d_q[0]-1)// Bi not enough, level is not largest non_empty
		//		{
		//
		//		}
		return 0;
	}
	//    BucketLine<Ktype>* getIthBucket(int i);
	//    SignalLine<Ktype>* getIthSignal(int i);
	__device__
	void bulkUpdate(Ktype k, int p)
	{

	}

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
	int getMaxPriorityOnBucket(int i, int q);

	int getNonEmptyBucketSignalIndex();

	void maintainNumBuckets();
	Ktype deleteMin();
	void deleteItem(Ktype k);

	void empty(int signalIndex);
	void fill(int bucketIndex);
};

template <class Ktype, class memspace=device_memspace>
class ParBucketHeap: public ParBucketHeapBase<Ktype>
{
public:
	int nodes,bulkSize,activeLvs;// n.d.q
	int max_levels, cuncurrent_levels;
	typedef ParBucketHeapBase<Ktype> parent_type;
	typename vector_type<int,memspace>::type locks_shared;
	typename vector_type<int,memspace>::type times_shared;
	typename vector_type<int,memspace>::type priorities_shared;
	typename vector_type<int,memspace>::type sigSizes_shared;
	typename vector_type<int,memspace>::type bucSizes_shared;
	typename vector_type<int,memspace>::type sigOffsets_shared;
	typename vector_type<int,memspace>::type bucOffsets_shared;
	typename vector_type<VoxBucketItem<Ktype>,memspace>::type buckets_shared;
	typename vector_type<VoxBucketItem<Ktype>,memspace>::type signals_shared;

	typename vector_type<int,memspace>::type dbg_shared;

	ParBucketHeap(int n_,int d_=1):nodes(n_),bulkSize(d_),activeLvs(-1),
			max_levels(log(n_/d_)/log(4)+1),cuncurrent_levels(ceil(1.0f*max_levels/2)),
			parent_type(n_,d_),locks_shared(cuncurrent_levels),
			times_shared(max_levels),priorities_shared(max_levels),
			sigSizes_shared(max_levels),bucSizes_shared(max_levels),
			sigOffsets_shared(max_levels),bucOffsets_shared(max_levels),
			dbg_shared(cuncurrent_levels)
	{
// consider use thrust::prefixsum..
		thrust::sequence(locks_shared.begin(),locks_shared.end(),0);
		thrust::sequence(times_shared.begin(),times_shared.end(),0);
		thrust::sequence(priorities_shared.begin(),priorities_shared.end(),0);
		thrust::sequence(sigSizes_shared.begin(),sigSizes_shared.end(),0);
		thrust::sequence(bucSizes_shared.begin(),bucSizes_shared.end(),0);
		bucOffsets_shared[0]=0;
		sigOffsets_shared[0]=0;
		for(int lv=1;lv<max_levels;lv++)
		{
				//			int bucketCapacity = d*pow(2, (2*id+1));     // Bi
				//			int signalCapacity = d*pow(2, (2*id)); // Si+1
				bucOffsets_shared[lv]=bucOffsets_shared[lv-1]+bulkSize*pow(2, (2*lv-1));  //lv=1,start from 2
				sigOffsets_shared[lv]=sigOffsets_shared[lv-1]+bulkSize*pow(2, (2*lv-2));  //lv=1.start from 1

		}
		int total_bucElems=bucOffsets_shared[max_levels-1]+bulkSize*pow(2, (2*max_levels+1));
		int total_sigElems=sigOffsets_shared[max_levels-1]+bulkSize*pow(2, (2*max_levels));
		buckets_shared.resize(total_bucElems);
		signals_shared.resize(total_sigElems);

		using thrust::raw_pointer_cast;
		this->d_locks=raw_pointer_cast(&locks_shared[0]);
		this->d_timestamps=raw_pointer_cast(&times_shared[0]);
		this->d_priorities=raw_pointer_cast(&priorities_shared[0]);
		this->d_active_signals=raw_pointer_cast(&sigSizes_shared[0]);
		this->d_active_buckets=raw_pointer_cast(&bucSizes_shared[0]);
		this->d_bucketOffsets=raw_pointer_cast(&bucOffsets_shared[0]);
		this->d_signalOffsets=raw_pointer_cast(&sigOffsets_shared[0]);
		this->voxBuckets=raw_pointer_cast(&buckets_shared[0]);
		this->voxSignals=raw_pointer_cast(&signals_shared[0]);

		this->d_mem=raw_pointer_cast(&dbg_shared[0]);


	}


};


