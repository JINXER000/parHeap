#include "parbucket.cuh"
#include "parDjikstra.h"
#include "utils.h"

#include "BucketHeap.h"
#include "BucketSignal.h"

namespace parheap{
template <class Ktype>
__global__
void BH_insertTest(ParBucketHeap<Ktype> bh,
		VoxBucketItem<Ktype>* eInVec,
		VoxBucketItem<Ktype>* eOutVec,
		int vec_num,bool *finished)
{
	const int level=blockIdx.x;
	const int thid=threadIdx.x;

	// assume push and pop vec_num elems
	if(level==0)
	{
		for(int i=0;i<vec_num;i++)
		{
			bh.updateRes(eInVec[i]);
		}
		for(int i=0;i<vec_num;i++)
		{
			bh.extractMinRes(eOutVec[i]);
		}
		*finished=true;
	}else
	{
		do
		{
			int isFail3=0;
			//resolve
			isFail3=bh.Resolve(level);  // can fail because !metConstrain

		}while(!*finished);

	}
}

template <class Ktype>
__global__
void BH_iter(ParBucketHeap<Ktype> bh,
		int inputSize, int numEdges, int numVertices,
		VoxBucketItem<Ktype>* srcNodes,
		int* distance,
		AdjacentNode* adjList,
		int* edgesOffset,
		int* edgesSize,
		bool* settled,
		bool* finished,
		Ktype destination)
{
	const int level=blockIdx.x;
	const int thid=threadIdx.x;

	// do update

	if(level==0)
	{
//		// insert all, initialize
		for(int iv=0;iv<numVertices;iv++) // iv: index of vertex
		{
			bool is_src=false;
			int init_priority=0;
			for(int is=0;is<inputSize;is++) // is: index of src
			{
				if(iv==srcNodes[is].key)
				{
					is_src=true;
					break;
				}
			}
			if(is_src)
			{
				init_priority=0;
			}else
			{
				init_priority=INT_MAX-1;
			}
			bh.updateRes(iv,init_priority);
			distance[iv]=init_priority;
		}
		VoxBucketItem<Ktype> eOut;
		for(int round=0;round<numVertices;round++)
		{
			bh.extractMinRes(eOut);
			// check finish
			if(eOut.key==destination)
			{
				*finished=true;
				break;
			}
			// mark v as settled
			Ktype v=eOut.key;
			int p=eOut.priority;
			settled[v]=1;

			for(int ie=edgesOffset[v];ie<edgesOffset[v]+edgesSize[v];ie++) // ie: index of edge
			{
				// if u is not settled, update((u,p+w))
				AdjacentNode v2u=adjList[ie];
				Ktype u=v2u.terminalVertex; // index of terminal vertex
				int w=v2u.weight;
				if(settled[u])
					continue;
				if(distance[u]>p+w)
				{
					distance[u]=p+w;
					bh.updateRes(u,p+w);
				}
			}
		}
	}
	else // level>0
	{
		do
		{
			int isFail3=0;
			//resolve
			isFail3=bh.Resolve(level);  // can fail because !metConstrain
		}while(!*finished);
	}
}


template <class Ktype>
__global__
void BH_insertSerail(ParBucketHeap<Ktype> bh,
		VoxBucketItem<Ktype>* eInPtr,
		int* test_vec)
{


	// do update


	int isFail=bh.update(*eInPtr);
	//resolve
	for (int level=0;level<bh.max_levels;level++)
	{
		if(!bh.metConstrain(level))
		{
			continue;
		}
		bh.ResSerial(level);
	}


}

template <class Ktype>
__global__
void BH_extractSerail(ParBucketHeap<Ktype> bh,
		int* miniElem)
{


	// do update

	VoxBucketItem<Ktype> eout;
	int isFail=bh.extractMin(eout);

	//resolve
	for (int level=0;level<bh.max_levels;level++)
	{
		if(!bh.metConstrain(level))
		{
			continue;
		}
		bh.ResSerial(level);
	}
	*miniElem=eout.key;


}



int parDijkstra(std::vector<int> &srcNode,
		Graph<AdjacentNode> &cuGraph,
		std::vector<int> &distances,
		int destination)
{
	using thrust::raw_pointer_cast;


	///  initCudaGraph
	int inputSize=srcNode.size();
	std::vector<VoxBucketItem<int>> h_srcNode(inputSize);
	thrust::device_vector<VoxBucketItem<int>> d_srcNode(inputSize);
	for(int id=0;id<inputSize;id++)
	{
		h_srcNode[id].setVal(srcNode[id],id%5);
	}

	thrust::copy(h_srcNode.begin(),h_srcNode.end(),d_srcNode.begin());

	thrust::device_vector<int> d_distance(cuGraph.numVertices);
	thrust::copy(distances.begin(),distances.end(),d_distance.begin());

	thrust::device_vector<AdjacentNode> d_adjLists(cuGraph.numEdges);
	thrust::device_vector<int> d_edgesOffset(cuGraph.numVertices);
	thrust::device_vector<int> d_edgesSize(cuGraph.numVertices);
	thrust::copy(cuGraph.adjacencyList.begin(),cuGraph.adjacencyList.end(),d_adjLists.begin());
	thrust::copy(cuGraph.edgesOffset.begin(),cuGraph.edgesOffset.end(),d_edgesOffset.begin());
	thrust::copy(cuGraph.edgesSize.begin(),cuGraph.edgesSize.end(),d_edgesSize.begin());

	thrust::device_vector<bool> d_settled(cuGraph.numVertices);

	// INIT BUCKET HEAP
	int nodes=inputSize;
	BucketHeap* bucketHeap = new BucketHeap();
	std::cout<<"input sources has "<<inputSize<<std::endl;

	ParBucketHeap<int> bh(nodes+2,1);
	int block_size=1;
	int grid_size=bh.max_levels;

	/// SERIAL TEST
//		thrust::device_vector<int> d_test_vec(inputSize);
//			for(int i=0;i<inputSize;i++)
//			{
//
//
//				BH_insertSerail<int><<<1,1>>>(bh,
//						raw_pointer_cast(&d_srcNode[i]),
//						raw_pointer_cast(&d_test_vec[0]));
//
//								bh.printAllItems();
//				//	    bucketHeap->update(h_srcNode[i].key,h_srcNode[i].priority);
//				//	    bucketHeap->printBucketCPU();
//			}
//
//			for(int i=0;i<inputSize;i++)
//			{
//				BH_extractSerail<int><<<1,1>>>(bh,
//						raw_pointer_cast(&d_test_vec[i]));
//
//				bh.printAllItems();
//				int out=d_test_vec[i];
//				printf("extracted min is %d \n",out);
//				int B0Size=bh.bucSizes_shared[0];
//				if(B0Size==0)
//					break;
//			}

	///// MUTEX TEST
		thrust::device_vector<VoxBucketItem<int>> d_outNodes(inputSize);
		bool *finished;
		CUDA_ALLOC_DEV_MEM(&finished,sizeof(int));
		CUDA_DEV_MEMSET(finished,0,sizeof(int));
		BH_insertTest<int><<<grid_size,block_size>>>(bh,
				raw_pointer_cast(&d_srcNode[0]),
				raw_pointer_cast(&d_outNodes[0]),
				nodes,finished);
		CUDA_FREE_DEV_MEM(finished);

		bh.printAllItems();

		for(int i=0;i<inputSize;i++)
		{
			VoxBucketItem<int> item=d_outNodes[i];
			std::cout<<"("<<item.key<<", "<<item.priority<<")";
		}



	///// Pardjikstra
//	ParBucketHeap<int> bh(cuGraph.numVertices,1);
//	int block_size=1;
//	int grid_size=bh.max_levels;
//
//	bool *finished;
//	CUDA_ALLOC_DEV_MEM(&finished,sizeof(int));
//	CUDA_DEV_MEMSET(finished,0,sizeof(int));
//	int* d_destination;
//	CUDA_ALLOC_DEV_MEM(&d_destination,sizeof(int));
//	CUDA_MEMCPY_H2D(d_destination,&destination,sizeof(int));
//
//
//	BH_iter<int><<<grid_size,block_size>>>(bh,
//			inputSize, cuGraph.numEdges, cuGraph.numVertices,
//			raw_pointer_cast(&d_srcNode[0]),
//			raw_pointer_cast(&d_distance[0]),
//			raw_pointer_cast(&d_adjLists[0]),
//			raw_pointer_cast(&d_edgesOffset[0]),
//			raw_pointer_cast(&d_edgesSize[0]),
//			raw_pointer_cast(&d_settled[0]),
//			 finished,
//			destination);
//	CUDA_FREE_DEV_MEM(d_destination);
//	CUDA_FREE_DEV_MEM(finished);
//
//	int dest_dist=d_distance[destination];
//	std::cout<<"finaldist= "<<dest_dist;


	return 0;
}
}

