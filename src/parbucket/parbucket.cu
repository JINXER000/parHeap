#include "parbucket.cuh"
#include "parDjikstra.h"
#include "utils.h"

#include "BucketHeap.h"
#include "BucketSignal.h"

template <class Ktype>
__global__
void BH_insertOnly(ParBucketHeap<Ktype> bh,
		VoxBucketItem<Ktype>* eInPtr,
		int* test_vec)
{
	const int level=blockIdx.x;
	const int thid=threadIdx.x;

	// do update

	if(level==0)
	{
		int isFail=bh.update(eInPtr);
		//resolve
		test_vec[level]=bh.Resolve(level);

	}else
	{
		//resolve
		test_vec[level]=bh.Resolve(level);
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
		int* test_vec)
{
	const int level=blockIdx.x;
	const int thid=threadIdx.x;

	// do update

	if(level==0)
	{
		VoxBucketItem<Ktype> *eInPtr;
		//		do
		//		{
		for(int is=0;is<inputSize;is++) // is: index of src
		{
			// mark v as settled
			Ktype v=srcNodes[is].key;
			int p=srcNodes[is].priority;
			settled[v]=1;
			// if u is not settled, update((u,p+w))
			for(int ie=edgesOffset[v];ie<edgesOffset[v]+edgesSize[v];ie++) // ie: index of edge
			{
				AdjacentNode v2u=adjList[ie];
				Ktype u=v2u.terminalVertex; // index of terminal vertex
				int w=v2u.weight;
				if(settled[u])
					continue;
				eInPtr->setVal(u,p+w);
				int isFail=bh.update(eInPtr);
				//resolve
				test_vec[level]=bh.Resolve(level);
			}

		}
		// extract elem with min priority

		VoxBucketItem<Ktype> eOut;
		int isFail=bh.extractMin(eOut);
		test_vec[level]=bh.Resolve(level);

		// for next round update
		eInPtr->setVal(eOut.key,eOut.priority);
		//		}while(bh.q>=0);


	}else
	{
		//resolve
		test_vec[level]=bh.Resolve(level);
	}



}


template <class Ktype>
__global__
void BH_insertSerail(ParBucketHeap<Ktype> bh,
		VoxBucketItem<Ktype>* eInPtr,
		int* test_vec)
{


	// do update


	int isFail=bh.update(eInPtr);
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
	if(isFail)
	{
		printf("wtf");
	}
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
		std::vector<int> &distances)
{


	BucketHeap* bucketHeap = new BucketHeap();

	// initCudaGraph
	int inputSize=srcNode.size();
	std::vector<VoxBucketItem<int>> h_srcNode(inputSize);
	thrust::device_vector<VoxBucketItem<int>> d_srcNode(inputSize);
	for(int id=0;id<inputSize;id++)
	{
		h_srcNode[id].setVal(srcNode[id],abs(id%5));
	}
	/// for input test
	//	std::vector<VoxBucketItem<int>> h_srcNode(inputSize);
	//	thrust::device_vector<VoxBucketItem<int>> d_srcNode(inputSize);
	//	for(int id=0;id<inputSize;id++)
	//	{
	//		h_srcNode[id].setVal(id,(id-inputSize/2)*(id-inputSize/2));
	//	}item.key
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

	// launch kernel
	int nodes=inputSize;
	ParBucketHeap<int> bh(nodes,1);
	std::cout<<"input sources has "<<inputSize<<std::endl;
	using thrust::raw_pointer_cast;
	thrust::device_vector<int> d_test_vec(3);
	int block_size=1;
	int grid_size=bh.max_levels;

	for(int i=0;i<inputSize;i++)
	{
		//		BH_insertOnly<int><<<grid_size,block_size>>>(bh,
		//				raw_pointer_cast(&d_srcNode[i]),
		//				raw_pointer_cast(&d_test_vec[0]));

		BH_insertSerail<int><<<1,1>>>(bh,
				raw_pointer_cast(&d_srcNode[i]),
				raw_pointer_cast(&d_test_vec[0]));

		//				bh.printAllItems();
		//	    bucketHeap->update(h_srcNode[i].key,h_srcNode[i].priority);
		//	    bucketHeap->printBucketCPU();
	}
	bh.printAllItems();
	int B0Size=bh.bucSizes_shared[0];
	while(B0Size>0)
	{
		BH_extractSerail<int><<<1,1>>>(bh,
				raw_pointer_cast(&d_test_vec[0]));

		bh.printAllItems();
		int out=d_test_vec[0];
		printf("extracted min is %d \n",out);
		B0Size=bh.bucSizes_shared[0];
	}
	// TODO perform V rounds
	//	BH_iter<int><<<grid_size,block_size>>>(bh,
	//			inputSize, cuGraph.numEdges, cuGraph.numVertices,
	//			raw_pointer_cast(&d_srcNode[0]),
	//			raw_pointer_cast(&d_distance[0]),
	//			raw_pointer_cast(&d_adjLists[0]),
	//			raw_pointer_cast(&d_edgesOffset[0]),
	//			raw_pointer_cast(&d_edgesSize[0]),
	//			raw_pointer_cast(&d_settled[0]),
	//			raw_pointer_cast(&d_test_vec[0]));

//	std::vector<int > h_test_vec(3);
//	thrust::copy(d_test_vec.begin(),d_test_vec.end(),h_test_vec.begin());
//
//	for(int i=0;i<3;i++)
//	{
//		std::cout<<h_test_vec[i]<<std::endl;
//
//	}

	return 0;
}


