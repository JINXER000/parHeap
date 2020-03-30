#include "parbucket.cuh"
#include "parDjikstra.h"
#include "utils.h"



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



int parDijkstra(std::vector<int> &srcNode,
		Graph<AdjacentNode> &cuGraph,
		std::vector<int> &distances)
{
	int nodes=16;

	ParBucketHeap<int> bh(nodes,1);
	// initCudaGraph
	int inputSize=srcNode.size();

	std::vector<VoxBucketItem<int>> h_srcNode(inputSize);
	thrust::device_vector<VoxBucketItem<int>> d_srcNode(inputSize);
	for(int id=0;id<inputSize;id++)
	{
		h_srcNode[id].setVal(srcNode[id],0);
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

	// launch kernel
	std::cout<<"input sources has "<<inputSize<<std::endl;
	using thrust::raw_pointer_cast;
	thrust::device_vector<int> d_test_vec(3);
	int block_size=1;
	int grid_size=bh.max_levels;


	// TODO perform V rounds
	BH_iter<int><<<grid_size,block_size>>>(bh,
			inputSize, cuGraph.numEdges, cuGraph.numVertices,
			raw_pointer_cast(&d_srcNode[0]),
			raw_pointer_cast(&d_distance[0]),
			raw_pointer_cast(&d_adjLists[0]),
			raw_pointer_cast(&d_edgesOffset[0]),
			raw_pointer_cast(&d_edgesSize[0]),
			raw_pointer_cast(&d_settled[0]),
			raw_pointer_cast(&d_test_vec[0]));

	std::vector<int > h_test_vec(3);
	thrust::copy(d_test_vec.begin(),d_test_vec.end(),h_test_vec.begin());

	for(int i=0;i<3;i++)
	{
		std::cout<<h_test_vec[i]<<std::endl;

	}

	return 0;
}


