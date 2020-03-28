#pragma once

#include <utils.h>
#include <cuda_utils.h>
#include <MinHeap.h>
#include "atomicLock.h"

typedef thrust::device_system_tag device_memspace;
typedef thrust::host_system_tag host_memspace;


template <typename T, typename M>
struct ptr_type {};

template <typename T>
struct ptr_type<T, host_memspace> {
  typedef T* type;
};
template <typename T>
struct ptr_type<T, device_memspace> {
  typedef thrust::device_ptr<T> type;
};
template <typename T, typename M>
struct vector_type {};

template <typename T>
struct vector_type<T, host_memspace> {
  typedef thrust::host_vector<T> type;
};
template <typename T>
struct vector_type<T, device_memspace> {
  typedef thrust::device_vector<T> type;
};

template <class Ktype>
struct __align__(16) VoxBucketItem {
	Ktype key;
	int priority;
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
//	__device__
//	void IncStamp()
//	{
//		d_timestamp[0]++;
//	}
//	__device__ int getTimeStamp()
//	{
//		int ts=d_timestamp[0];
//		return ts;
//	}
//
//
//};


template <class Ktype>
struct ParBucketHeapBase
{

	// param for the whole heap
	int q; // largest non-empty level of Bi
	int max_level;// maximum level size
	int d; //bulk size

	int *d_q; // largest non-empty level of Bi
	int *d_max_level;// maximum level size
	int *d_d; //bulk size

	// param for each level
	// length==levels

	int *h_bucketCapacityOffsets;
	int *h_signalCapacityOffsets;
	int* h_priorities;
	int *h_active_buckets;
	int *h_active_signals;
	int *h_timestamps;

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

		max_level=log(n/d)/log(4)+1;
		//		for(int i=0;i<max_level;i++)
		//		{
		//			new (&bs_levels[i]) BSlevel<Ktype>(i,d);
		//		}



	}

	void creadAllMem()
	{
		int cuncurrent_levels=ceil(1.0f*max_level/2);
		CUDA_ALLOC_DEV_MEM(&d_locks,sizeof(int)*cuncurrent_levels);
		CUDA_ALLOC_DEV_MEM(&d_mem,sizeof(int)*cuncurrent_levels);
		CUDA_DEV_MEMSET(d_mem,0,sizeof(int)*cuncurrent_levels);
		param_initD();
		param_H2D();
		levelParamInitD();
		levelParamH2D();


	}
	void deleteAllMem()
	{
		//		delete [] bs_levels;

		CUDA_FREE_DEV_MEM(d_mem);
		param_deinitD();
		levelParamDeinitD();
		CUDA_FREE_DEV_MEM(d_locks);
	}
	void levelParamInitD()
	{
		CUDA_ALLOC_DEV_MEM(&d_bucketOffsets,sizeof(int)*max_level);
		CUDA_ALLOC_DEV_MEM(&d_signalOffsets,sizeof(int)*max_level);
		CUDA_ALLOC_DEV_MEM(&d_priorities,sizeof(int)*max_level);
		CUDA_ALLOC_DEV_MEM(&d_active_signals,sizeof(int)*max_level);
		CUDA_ALLOC_DEV_MEM(&d_active_buckets,sizeof(int)*max_level);
		CUDA_ALLOC_DEV_MEM(&d_timestamps,sizeof(int)*max_level);

		h_bucketCapacityOffsets=new int[max_level];
		h_signalCapacityOffsets=new int[max_level];
		h_priorities=new int[max_level];
		h_active_signals=new int[max_level];
		h_active_buckets=new int[max_level];
		h_timestamps=new int[max_level];

		h_bucketCapacityOffsets[0]=0;
		h_signalCapacityOffsets[0]=0;
		for(int lv=0;lv<max_level;lv++)
		{
			h_priorities[lv]=0;
			h_timestamps[lv]=0;
			h_active_buckets[lv]=0;
			h_active_signals[lv]=0;


			if(lv>0)
			{
				//			int bucketCapacity = d*pow(2, (2*id+1));     // Bi
				//			int signalCapacity = d*pow(2, (2*id)); // Si+1
				h_bucketCapacityOffsets[lv]=h_bucketCapacityOffsets[lv-1]+d*pow(2, (2*lv-1));
				h_signalCapacityOffsets[lv]=h_signalCapacityOffsets[lv-1]+d*pow(2, (2*lv-2));
			}


		}
	}
	__host__ void levelParamD2H()
	{


		CUDA_MEMCPY_D2H(h_bucketCapacityOffsets,d_bucketOffsets,sizeof(int)*max_level);
		CUDA_MEMCPY_D2H(h_priorities,d_priorities,sizeof(int)*max_level);
		CUDA_MEMCPY_D2H(h_signalCapacityOffsets,d_signalOffsets,sizeof(int)*max_level);
		CUDA_MEMCPY_D2H(h_active_signals,d_active_signals,sizeof(int)*max_level);
		CUDA_MEMCPY_D2H(h_active_buckets,d_active_buckets,sizeof(int)*max_level);
		CUDA_MEMCPY_D2H(h_timestamps,d_timestamps,sizeof(int)*max_level);
	}
	__host__ void levelParamH2D()
	{
		CUDA_MEMCPY_H2D(d_bucketOffsets,h_bucketCapacityOffsets,sizeof(int)*max_level);
		CUDA_MEMCPY_H2D(d_priorities,h_priorities,sizeof(int)*max_level);
		CUDA_MEMCPY_H2D(d_signalOffsets,h_signalCapacityOffsets,sizeof(int)*max_level);
		CUDA_MEMCPY_H2D(d_active_signals,h_active_signals,sizeof(int)*max_level);
		CUDA_MEMCPY_H2D(d_active_buckets,h_active_buckets,sizeof(int)*max_level);
		CUDA_MEMCPY_H2D(d_timestamps,h_timestamps,sizeof(int)*max_level);
	}
	__host__ void levelParamDeinitD()
	{
		CUDA_FREE_DEV_MEM(d_bucketOffsets);
		CUDA_FREE_DEV_MEM(d_priorities);
		CUDA_FREE_DEV_MEM(d_signalOffsets);
		CUDA_FREE_DEV_MEM(d_active_signals);
		CUDA_FREE_DEV_MEM(d_active_buckets);
		CUDA_FREE_DEV_MEM(d_timestamps);

		delete [] h_bucketCapacityOffsets;
		delete [] h_signalCapacityOffsets;
		delete [] h_priorities;
		delete [] h_active_signals;
		delete [] h_active_buckets;
		delete [] h_timestamps;
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
		checkCudaErrors(cudaFree(d_q));
		//		CUDA_FREE_DEV_MEM(d_q);
		CUDA_FREE_DEV_MEM(d_max_level);
		CUDA_FREE_DEV_MEM(d_d);
	}

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
		if(level==d_q[0]-1)
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
	int n,d,q;
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

	ParBucketHeap(int n_,int d_=1):n(n_),d(d_),q(-1),
			max_levels(log(n/d)/log(4)+1),cuncurrent_levels(ceil(1.0f*max_levels/2)),
			parent_type(n,d),locks_shared(cuncurrent_levels),
			times_shared(max_levels),priorities_shared(max_levels),
			sigSizes_shared(max_levels),bucSizes_shared(max_levels),
			sigOffsets_shared(max_levels),bucOffsets_shared(max_levels)
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
				bucOffsets_shared[lv]=bucOffsets_shared[lv-1]+d*pow(2, (2*lv-1));
				sigOffsets_shared[lv]=sigOffsets_shared[lv-1]+d*pow(2, (2*lv-2));

		}

		using thrust::raw_pointer_cast;
		this->d_locks=raw_pointer_cast(&locks_shared[0]);
		this->d_timestamps=raw_pointer_cast(&times_shared[0]);
		this->d_priorities=raw_pointer_cast(&priorities_shared[0]);
		this->d_active_signals=raw_pointer_cast(&sigSizes_shared[0]);
		this->d_active_buckets=raw_pointer_cast(&bucSizes_shared[0]);
		this->d_bucketOffsets=raw_pointer_cast(&bucOffsets_shared[0]);
		this->d_signalOffsets=raw_pointer_cast(&sigOffsets_shared[0]);


	}


};


