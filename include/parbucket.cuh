#pragma once

#include <utils.h>
#include <cuda_utils.h>
#include <MinHeap.h>
#include "atomicLock.h"


namespace parheap{
template <class Ktype>
struct __align__(16) VoxBucketItem {
	Ktype key;
	int priority;

	__device__ __host__
	void setVal(Ktype k,int p)
	{
		key=k;
		priority=p;
	}
};


template <class Ktype>
__global__ void predicate(VoxBucketItem<Ktype> *Bi, int szBi, int* nonEqual)
{
	int thid=blockIdx.x*blockDim.x+threadIdx.x;
	if(thid==0)
	{
		nonEqual[0]=1;
	}
	else if(thid<szBi)
	{
		int this_key=Bi[thid].key;
		int last_key=Bi[thid-1].key;
		nonEqual[thid]=(this_key==last_key)?0:1;
	}
}
template <class Ktype>
__global__ void predicateMLB(VoxBucketItem<Ktype> *Bi, int szBi, int* pred, int priority)
{
	int thid=blockIdx.x*blockDim.x+threadIdx.x;
	if(thid<szBi)
	{
		pred[thid]=(Bi[thid].priority<=priority)?0:1;
	}
}

template<typename Ktype>
__global__
void moveElemShM(VoxBucketItem<Ktype>* oldAddr, VoxBucketItem<Ktype>* newAddr,
		int* d_scan, int* d_pred, int numElems)
{
	extern __shared__ VoxBucketItem<Ktype> inputBuffer[];
	int gid=threadIdx.x+blockIdx.x*blockDim.x;
	int thid=threadIdx.x;
	if(gid>=numElems)
		return;

	inputBuffer[thid].setVal(oldAddr[gid].key,oldAddr[gid].priority);
	__syncthreads();

	int pos;
	if(d_pred[gid]==1)
	{
		pos=d_scan[gid];
	}else
		return;
	if(pos>numElems)
		return;
	// if pos is in another block, this will result in problem...
	//	while(Bi[pos].copied!=true)
	//	{
	//		printf("wtf???? block not in order");
	//	}
	newAddr[pos].setVal(inputBuffer[thid].key,inputBuffer[thid].priority);
}


template <class Ktype>
struct ParBucketHeapBase
{

	// param for the whole heap
	int *q; // largest non-empty level of Bi
	int d; //bulk size
	int maxLV;

	bool operationOK;

	// param for each level
	// length==levels


	int *d_bucketOffsets;
	int *d_signalOffsets;
	int* d_priorities;
	int *d_active_buckets;
	int *d_active_signals;
	int *d_max_buckets;
	int *d_max_signals;
	int *d_timestamps;

	// mutex,length==levels/2
	int* d_locks;
	int* d_mem;


	// buckets and signals
	VoxBucketItem<Ktype> *bucketSignals;

	//	BSlevel<Ktype>* bs_levels;
	ParBucketHeapBase(int n,int d_=1):d(d_),operationOK(0)
	{
		maxLV=ceil(log(n/d)/log(4));
		printf("max level is %d \n", maxLV);
	}

	__device__
	VoxBucketItem<Ktype> *getBucItem(int lv,int id)
	{
		int glbId=d_bucketOffsets[lv]+id;
		return &(bucketSignals[glbId]);
	}

	__device__
	VoxBucketItem<Ktype> *getSigItem(int lv,int id)
	{
		int glbId=d_signalOffsets[lv]+id;
		return &(bucketSignals[glbId]);
	}

	__device__
	void updateRes(Ktype k, int p)
	{
		VoxBucketItem<Ktype> eIn;
		eIn.setVal(k,p);
		updateRes(eIn);
	}

	__device__
	void updateRes(VoxBucketItem<Ktype> eIn)
	{
		int isFail1=0,isFail2=0;
		do{

			isFail1=update(eIn);  // can fail because not yet resolved
			//resolve
			isFail2=Resolve(0);  // can fail because !metConstrain

		}while(isFail1<0||isFail2<0);
		// only fail 1, then resolve can address
		// only fail 2, then 1 will fail as well.
		// fail both because lv 2 takes too long
	}
	__device__
	int update(VoxBucketItem<Ktype> eIn)
	{
		if(!metConstrain(0))
			return -1;


		if(d_active_signals[0]>0)
		{
			printf("update precondition not met!");
			assert(false);
			return -1;
		}
		VoxBucketItem<Ktype>* S0=getSigItem(0,0);
		S0->setVal(eIn.key,eIn.priority);
		d_active_signals[0]++;
		operationOK=true;
		return 0;
	}
	__device__ int  findMin()
	{
		// TODO reduce
		int minPriority=INT_MAX;
		int idOut=0;
		for(int i=0;i<d_active_buckets[0];i++)
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
	__device__ void extractMinRes(VoxBucketItem<Ktype> &eOut)
	{
		int isFail1=0,isFail2=0;
		do{

			isFail1=extractMin(eOut);  // can fail because not yet resolved
			//resolve
			isFail2=Resolve(0);  // can fail because !metConstrain

		}while(isFail1<0||isFail2<0);
		// only fail 1, then resolve can address
		// only fail 2, then 1 will fail as well.
		// fail both because lv 2 takes too long
	}
	__device__ int  extractMin(VoxBucketItem<Ktype> &eOut)
	{
		if(!metConstrain(0))
			return -1;


		if(d_active_buckets[0]<d || *q<0  )
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

		//				VoxBucketItem<Ktype>* S0=getSigItem(0,0);
		//				S0[0].setVal(eOutPtr->key,-1 );
		//				d_active_signals[0]++;

		eOut.setVal(eOutPtr->key,eOutPtr->priority);
		removeElemB(0,idOut);
		operationOK=true;
		return 0;
	}

	__device__ void removeElemB(int lv,int id)
	{
		VoxBucketItem<Ktype>* Bi=getBucItem(lv,0);
		for(int i=id;i<d_active_buckets[lv];i++)
		{
			Bi[i].setVal(Bi[i+1].key,Bi[i+1].priority);
		}
		d_active_buckets[lv]--;
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
			if(4*(next_cnt+1)>this_cnt) // if this_cnt++, then not satisfy
				return true;
			else
				return false;
		}
		//  last level
		if(level==*q)
		{
			last_cnt=getTimeStamp(level-1);
			if( last_cnt==4*(this_cnt+1) )
				return true;
			else
				return false;
		}//else
		next_cnt=getTimeStamp(level+1);
		last_cnt=getTimeStamp(level-1);
		if(4*(next_cnt+1)>this_cnt  && last_cnt==4*(this_cnt+1))
			return true;
		else
			return false;
	}


	__device__
	void ResSerial(int level)
	{
		IncStamp(level);
		if(level>0&&level>*q)
			return;


		int sigS=d_active_signals[level];
		int bucS=d_active_buckets[level];
		///////////////////////
		//		 empty if needed
		if(d_active_signals[level]>0)
		{

			// Bi<-merge(Si,Bi)
			VoxBucketItem<Ktype>* Bi=getBucItem(level,0);
			VoxBucketItem<Ktype>* Si=getSigItem(level,0);



			Merge(Si,Bi,d_active_signals[level],d_active_buckets[level]);
			clearS(Si,d_active_signals[level]);
			DelDupOnBucket(level);

			int p_i_old=getMaxPriorityOnBucket(level);  // MAY BE SLOW
			int num=smallerPonBucket(level,p_i_old);
			int properSize=d_max_buckets[level];
			if(num>properSize)
			{

				// return num actually
				SelectP(level,properSize,d_priorities[level]);
				// Bi_dot={e.p>p_i}
				VoxBucketItem<Ktype>* Bi_dot=getBucItem(level,properSize);


				VoxBucketItem<Ktype>* Siplus1=getSigItem(level+1,0);
				Merge(Bi_dot,Siplus1,d_active_buckets[level]-properSize,d_active_signals[level+1]);
				// Bi = {e.p<=p_i}
				d_active_buckets[level]=properSize;
			}
			maintainPriorityOn(Bi,level);


		}
		//		NonEmptyBucketSignal(level);
		// fill if needed
		int BiSize=d_active_buckets[level];
		if(BiSize<d_max_buckets[level]&&level<*q)// Bi not enough, level is not largest non_empty
		{

			VoxBucketItem<Ktype>* BiPlus1=getBucItem(level+1,0);
			VoxBucketItem<Ktype>* SiPlus1=getSigItem(level+1,0);

			Merge(SiPlus1,BiPlus1,d_active_signals[level+1],d_active_buckets[level+1]);
			clearS(SiPlus1,d_active_signals[level+1]);
			DelDupOnBucket(level+1);
			maintainPriorityOn(BiPlus1,level+1);

			int properSize=d_max_buckets[level]-BiSize;

			int numFill=SelectP(level+1,properSize,d_priorities[level]);

			//BiPlus1_dot=={e in Bi+1,e.p<=pi}
			VoxBucketItem<Ktype>* BiPlus1_dot=getBucItem(level+1,0);
			VoxBucketItem<Ktype>* Bi=getBucItem(level,0);
			Merge(BiPlus1_dot,Bi,numFill,d_active_buckets[level]);
			maintainPriorityOn(Bi,level);
			//Biplus1=={e in Bi+1,e.p>pi}
			d_active_buckets[level+1]-=numFill;
			VoxBucketItem<Ktype>* Biplus1_tmp=getBucItem(level+1,numFill);
			moveItem(Biplus1_tmp,BiPlus1,d_active_buckets[level+1]);
			maintainPriorityOn(BiPlus1,level+1);
			// Si+1 <---- {e in Bi+1, e.p>pi+1}
			MoveLargerBack(BiPlus1,SiPlus1,d_priorities[level+1],
					d_active_buckets[level+1],d_active_signals[level+1]);



		}
		//////////// routine
		// maintain the largest non empty level, see report p5
		NonEmptyBucketSignal(level);
		//				printD(level);


	}

	__device__
	int Resolve(int level)
	{
		//+++++++++++++++++++++
		if(level>=maxLV)
			return -1;

		if(!metConstrain(level))
		{
			return -1;
		}

		LockSet<3> lockset;
		bool acuiredThis=false,acuiredLast=false, acuiredNext=false;
		bool acuiredAll=false;
		while(!acuiredAll)
		{
			acuiredThis=lockset.TryLock(d_locks[level]);
			if(level-1>=0)
				acuiredLast=lockset.TryLock(d_locks[level-1]);
			else
				acuiredLast=true;
			if(level+1<maxLV)
				acuiredNext=lockset.TryLock(d_locks[level+1]);
			else
				acuiredNext=true;

			acuiredAll=acuiredThis&&acuiredLast&&acuiredNext;
			if(!acuiredAll)
				lockset.YieldAll();
		}
		ResSerial(level);

		return d_timestamps[level];
	}
	__device__
	void MoveLargerBack(VoxBucketItem<Ktype>* BAddr,VoxBucketItem<Ktype>* SAddr,
			int pri,int &Bsize,int &Ssize)
	{
		int size=Bsize;
		int j=0;
		for(int i=0;i<Bsize;i++)
		{
			if(BAddr[i].priority>pri)
			{
				SAddr[j++].setVal(BAddr[i].key,BAddr[i].priority);
				size--;
			}
		}
		Bsize=size;
		Ssize=j;
	}
	__device__
	void MoveLargerBackPar(VoxBucketItem<Ktype>* BAddr,VoxBucketItem<Ktype>* SAddr,
			int pri,int &Bsize,int &Ssize)
	{

		if(Bsize<=1)
			return;

		int *pred=(int*)malloc(sizeof(int)*Bsize);
		if(pred==NULL)
			assert(false); // insufficient mem
		int blk_size=256;
		int grid_size=(Bsize%blk_size==0)?(Bsize/blk_size):(Bsize/blk_size+1);//ceil(1.0f*szBi/blk_size);
		predicateMLB<<<grid_size,blk_size>>>(BAddr,Bsize,pred,pri);

		int *d_scan=(int*)malloc(sizeof(int)*Bsize);
		if(d_scan==NULL)
			assert(false); // insufficient mem

		int finalSize=PrefixSum(d_scan,pred,Bsize);

		moveElemSerial(BAddr,SAddr,d_scan,pred,Bsize);
		//		moveElemShM<<<grid_size,blk_size,sizeof(VoxBucketItem<Ktype>)*blk_size>>>(BAddr,SAddr,d_scan,pred,Bsize);

		Ssize=finalSize;
		Bsize=Bsize-Ssize;

		free(pred);
		free(d_scan);
	}
	__device__
	void moveElemSerial(VoxBucketItem<Ktype>* oldAddr,VoxBucketItem<Ktype>* newAddr,
			int* d_scan, int* d_pred,int size)
	{
		for(int i=0;i<size;i++)
		{
			int pos;
			if(d_pred[i]==1)
				pos=d_scan[i];
			else if(d_pred[i]==0)
				continue;
			else
				assert(false);
			newAddr[pos].setVal(oldAddr[i].key,oldAddr[i].priority); // pos undefined, scan err
		}
	}
	__device__
	void moveItem(VoxBucketItem<Ktype>* oldAddr,VoxBucketItem<Ktype>* newAddr, int size)
	{
		// assume new is in front of old, or two array without overlap
		for(int i=0;i<size;i++)
		{
			newAddr[i].setVal(oldAddr[i].key,oldAddr[i].priority);
		}
	}
	__device__
	int getMaxPriorityOnBucket(int lv)
	{
		if(lv==*q&&d_active_signals[lv]==0)
			return INT_MAX;
		if(d_active_buckets[lv]==0)
			return INT_MIN;

		VoxBucketItem<Ktype> *Bi=getBucItem(lv,0);
		int maxPriority=INT_MIN;
		for (int i = 0; i < d_active_buckets[lv] ; i++ ){
			if (Bi[i].priority > maxPriority){
				maxPriority = Bi[i].priority;
			}
		}
		// try below
		//	    maxPriority=Bi[d_active_buckets[lv]-1].priority;
		d_priorities[lv]=maxPriority;
		return maxPriority;
	}
	__device__
	void DelDupOnBucket(int lv)
	{
				if(lv>=4)
					DelDupOnBucketPar(lv);
				else
		DelDupOnBucketSerail(lv);
	}
	__device__
	void DelDupOnBucketSerail(int lv)
	{
		// identify, delete, compress--- scan and prefix sum

		int &szBi=d_active_buckets[lv];
		if(szBi<=1)
			return;
		VoxBucketItem<Ktype> *Bi=getBucItem(lv,0);
		int i=0,j=1;
		while(j<szBi)
		{
			if(Bi[i].key!=Bi[j].key)
			{
				Bi[++i]=Bi[j++];
			}else
			{
				// same key, remove the one with larger p
				if(Bi[i].priority>Bi[j].priority)
					Bi[i].priority=Bi[j].priority;
				j++;

			}
		}
		// new active size of Bi= last idx+1
		szBi=i+1;
	}
	__device__
	void DelDupOnBucketPar(int lv)
	{
		// identify, delete, compress--- scan and prefix sum

		int &szBi=d_active_buckets[lv];
		if(szBi<=1)
			return;
		VoxBucketItem<Ktype> *Bi=getBucItem(lv,0);

		int *pred=(int*)malloc(sizeof(int)*szBi);
		if(pred==NULL)
			assert(false); // insufficient mem
		int blk_size=256;
		int grid_size=(szBi%blk_size==0)?(szBi/blk_size):(szBi/blk_size+1);//ceil(1.0f*szBi/blk_size);
		predicate<<<grid_size,blk_size>>>(Bi,szBi,pred);

		int *d_scan=(int*)malloc(sizeof(int)*szBi);
		if(d_scan==NULL)
			assert(false); // insufficient mem
		int test1=d_scan[0];
		int test2=pred[0];
		int finalSize=PrefixSum(d_scan,pred,szBi);

		//		moveElemInplace<int><<<grid_size,blk_size>>>(Bi,d_scan,pred,szBi);
		moveElemSerial(Bi,Bi,d_scan,pred,szBi);
		//		moveElemShM<<<grid_size,blk_size,sizeof(VoxBucketItem<Ktype>)*blk_size>>>
		//				(Bi,Bi,d_scan,pred,szBi);

		d_active_buckets[lv]=finalSize;
		free(pred);
		free(d_scan);
	}

	__device__
	void clearS(VoxBucketItem<Ktype> *Si,int &szSi)
	{
		// after modify the sz, all will be trash
		//		for(int i=0;i<szSi;i++)
		//		{
		//			Si[i].setVal(0,0);
		//		}
		szSi=0;
	}

	__device__
	void Merge(VoxBucketItem<Ktype> *itemIn, VoxBucketItem<Ktype> *itemOut,
			int szIn,int &szOut)// if needed, please modify zsIn out of the function
	{
		// thrust::merge
		// we assume 2 arrays are sorted

		if(szIn<=0)
			return;
		if(szOut<=0)
		{
			for(int i=0;i<szIn;i++)
			{
				// copy in to out
				itemOut[i]=itemIn[i];
			}
			//modify active size
			szOut=szIn;
			return;
		}
		int pin=szIn-1, pout=szOut-1;
		while(pin>=0&&pout>=0)
		{
			// sort by p, put the larger one at the very end
			if(itemOut[pout].priority>itemIn[pin].priority)
			{
				itemOut[pout+pin+1]=itemOut[pout];
				pout--;
			}else
			{
				itemOut[pout+pin+1]=itemIn[pin];
				pin--;
			}
		}
		while(pin>=0)
		{
			itemOut[pout+pin+1]=itemIn[pin];
			pin--;
		}
		//modify active size
		szOut=szOut+szIn;
		// cannot set szIn=0.
	}
	__device__
	void bulkUpdate(Ktype k, int p)
	{

	}
	// for wavefront init
	__device__
	void fillZero(VoxBucketItem<Ktype>* initArr, int size)
	{
		int bucLV=0;
		int bucId=0;
		VoxBucketItem<Ktype>* Bi= getBucItem(0,0);
		for(int i=0;i<size;i++)
		{
			Bi[bucId].setVal(initArr[i].key,0);
			bucId++;
			d_active_buckets[bucLV]=bucId;
			if(bucId>=d_max_buckets[bucLV])
			{
				bucLV++;
				bucId=0;
				Bi= getBucItem(bucLV,0);
				*q=bucLV;
				printD(bucLV);
			}
		}
	}
	__device__
	int smallerPonBucket(int lv,int maxP)
	{
		// count , proposition+prefix sum
		VoxBucketItem<Ktype>* Bi=getBucItem(lv,0);
		int num_smaller=0;
		for(int i=0;i<d_active_buckets[lv];i++)
		{
			if(Bi[i].priority>maxP)
				break;
			num_smaller++;
		}

		return num_smaller;
	}
	__device__
	int SelectP(int lv,int smallerNum,int &pi_out)
	{
		// since Bi is sorted, NO NEED to quick select??
		VoxBucketItem<Ktype>* Bi=getBucItem(lv,0);
		if(d_active_buckets[lv]<smallerNum)
		{
			pi_out=Bi[d_active_buckets[lv]-1].priority;
			//			assert(false);
			return d_active_buckets[lv];
		}
		pi_out=Bi[smallerNum-1].priority;
		return smallerNum;
	}

	__device__ void IncStamp(int lv)
	{
		if(lv==0&&operationOK==false)
		{
			// this resolve is for the last operation
			return;
		}
		d_timestamps[lv]++;
		operationOK=false;
	}
	__device__
	void maintainPriorityOn(VoxBucketItem<Ktype>* BucAddr,int lv)
	{
		int maxP=BucAddr[d_active_buckets[lv]-1].priority;
		d_priorities[lv]=maxP;
	}
	__device__
	void NonEmptyBucketSignal(int level)
	{
		// reduction
		int this_sigs=d_active_signals[level];
		int next_sigs=d_active_signals[level+1];
		int this_bucs=d_active_buckets[level];
		int next_bucs=d_active_buckets[level+1];

		if(next_sigs<=0&&next_bucs<=0)
		{
			if(*q==level+1)
			{
				*q=level;
			}
		}else
		{
			if(*q<level+1)
			{
				*q=level+1;
			}
		}

		if(this_sigs<=0&&this_bucs<=0)
		{
			if(*q==level)
			{
				*q=level-1;
			}
		}
		else
		{
			if(*q<level)
			{
				*q=level;
			}
		}

	}

	__device__ void printD(int level)
	{

		printf("\nThe current timestamp of level 0 is %d\n",d_timestamps[0]);
		for(int lv=0;lv<=*q;lv++)
		{
			printf("res level %d S%d\t:",level,lv);
			VoxBucketItem<Ktype>* Si=getSigItem(lv,0);
			for(int i=0;i<d_active_signals[lv];i++)
			{
				printf("(%d,%d)",Si[i].key,Si[i].priority);
			}
			printf("\n");
			printf("res level %d B%d\t:",level,lv);
			VoxBucketItem<Ktype>* Bi=getBucItem(lv,0);
			for(int i=0;i<d_active_buckets[lv];i++)
			{
				printf("(%d,%d)",Bi[i].key,Bi[i].priority);
			}
			printf("\n");

		}
		__threadfence_system();
	}

};

template <class Ktype, class memspace=device_memspace>
class ParBucketHeap: public ParBucketHeapBase<Ktype>
{
public:
	int nodes,bulkSize,activeLvs;// n.d.q
	int max_levels;
	typedef ParBucketHeapBase<Ktype> parent_type;
	typename vector_type<int,memspace>::type activeLVs_shared;
	typename vector_type<int,memspace>::type locks_shared;
	typename vector_type<int,memspace>::type times_shared;
	typename vector_type<int,memspace>::type priorities_shared;
	typename vector_type<int,memspace>::type sigSizes_shared;
	typename vector_type<int,memspace>::type bucSizes_shared;
	typename vector_type<int,memspace>::type maxSigSizes_shared;
	typename vector_type<int,memspace>::type maxBucSizes_shared;
	typename vector_type<int,memspace>::type sigOffsets_shared;
	typename vector_type<int,memspace>::type bucOffsets_shared;
	typename vector_type<VoxBucketItem<Ktype>,memspace>::type bucketSignals_shared;
	typename vector_type<int,memspace>::type dbg_shared;

	ParBucketHeap(int n_,int d_=1):nodes(n_),bulkSize(d_),activeLvs(-1),activeLVs_shared(1),
			max_levels(ceil(log(n_/d_)/log(4))),
			parent_type(n_,d_),locks_shared(max_levels+1),
			times_shared(max_levels+1),priorities_shared(max_levels+1),
			sigSizes_shared(max_levels+1),bucSizes_shared(max_levels+1),
			maxSigSizes_shared(max_levels+1),maxBucSizes_shared(max_levels+1),
			sigOffsets_shared(max_levels+1),bucOffsets_shared(max_levels+1),
			dbg_shared(max_levels+1)
	{
		//		parent_type.maxLV=max_levels;
		// consider use thrust::prefixsum..
		thrust::fill(locks_shared.begin(),locks_shared.end(),0);
		thrust::fill(times_shared.begin(),times_shared.end(),0);
		thrust::fill(priorities_shared.begin(),priorities_shared.end(),0);
		thrust::fill(sigSizes_shared.begin(),sigSizes_shared.end(),0);
		thrust::fill(bucSizes_shared.begin(),bucSizes_shared.end(),0);
		thrust::fill(dbg_shared.begin(),dbg_shared.end(),0);

		sigOffsets_shared[0]=0;
		bucOffsets_shared[0]=2;
		maxSigSizes_shared[0]=bulkSize;
		maxBucSizes_shared[0]=2*bulkSize;
		for(int lv=1;lv<max_levels;lv++)
		{
			maxSigSizes_shared[lv]=bulkSize*pow(2,2*lv);
			maxBucSizes_shared[lv]=bulkSize*pow(2,2*lv+1);
			// however, we double the size for temporal flow like Merge()

			// sig offset(current) = buck offset(last) +bucSize(last)
			sigOffsets_shared[lv]=bucOffsets_shared[lv-1]+maxBucSizes_shared[lv-1]*2;  //lv=1.start from 6
			bucOffsets_shared[lv]=sigOffsets_shared[lv]+maxSigSizes_shared[lv]*2;  //lv=1,start from 14

		}
		int total_elems=bucOffsets_shared[max_levels-1]+bulkSize*pow(2, 2*max_levels-1);
		bucketSignals_shared.resize(total_elems);

		using thrust::raw_pointer_cast;
		this->d_locks=raw_pointer_cast(&locks_shared[0]);
		this->d_timestamps=raw_pointer_cast(&times_shared[0]);
		this->d_priorities=raw_pointer_cast(&priorities_shared[0]);
		this->d_active_signals=raw_pointer_cast(&sigSizes_shared[0]);
		this->d_active_buckets=raw_pointer_cast(&bucSizes_shared[0]);
		this->d_bucketOffsets=raw_pointer_cast(&bucOffsets_shared[0]);
		this->d_signalOffsets=raw_pointer_cast(&sigOffsets_shared[0]);
		this->bucketSignals=raw_pointer_cast(&bucketSignals_shared[0]);
		this->q=raw_pointer_cast(&activeLVs_shared[0]);
		this->d_mem=raw_pointer_cast(&dbg_shared[0]);
		this->d_max_signals=raw_pointer_cast(&maxSigSizes_shared[0]);
		this->d_max_buckets=raw_pointer_cast(&maxBucSizes_shared[0]);


	}
	int getSigLoc(int lv, int id)
	{
		return sigOffsets_shared[lv]+id;
	}
	int getBucLoc(int lv, int id)
	{
		return bucOffsets_shared[lv]+id;
	}
	void resetParam()
	{
		activeLVs_shared[0]=-1;
		thrust::fill(locks_shared.begin(),locks_shared.begin()+max_levels,0);
		thrust::fill(times_shared.begin(),times_shared.begin()+max_levels,0);
		thrust::fill(priorities_shared.begin(),priorities_shared.begin()+max_levels,0);
		thrust::fill(sigSizes_shared.begin(),sigSizes_shared.begin()+max_levels,0);
		thrust::fill(bucSizes_shared.begin(),bucSizes_shared.begin()+max_levels,0);
	}
	void printAllItems()
	{
		activeLvs=activeLVs_shared[0];
		printf("====max active lv is%d====\n",activeLvs);

		for(int lv=0;lv<=activeLvs;lv++)
		{
			std::cout<<"S"<<lv<<"\t:";
			for(int i=0;i<sigSizes_shared[lv];i++)
			{
				int itemLoc=getSigLoc(lv,i);
				VoxBucketItem<Ktype> item=bucketSignals_shared[itemLoc];
				std::cout<<"("<<item.key<<", "<<item.priority<<")";
			}
			std::cout<<std::endl;
			std::cout<<"B"<<lv<<"\t:";
			for(int i=0;i<bucSizes_shared[lv];i++)
			{
				int itemLoc=getBucLoc(lv,i);
				VoxBucketItem<Ktype> item=bucketSignals_shared[itemLoc];
				std::cout<<"("<<item.key<<", "<<item.priority<<")";
			}
			std::cout<<std::endl;

		}
	}
};
}


