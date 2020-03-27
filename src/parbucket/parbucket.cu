#include "parbucket.h"



__global__
void BH_iter(ParBucketHeap<int3> bh,int* test_vec)
{
	int level=blockIdx.x;
	int thid=threadIdx.x;


	// do update

	//resolve
	test_vec[level]=bh.Resolve(level);


}



int parDijkstra()
{
	int nodes=16;
	ParBucketHeap<int3> bh(nodes,1);
	thrust::device_vector<int> d_test_vec(3);
	int block_size=1;
	int grid_size=bh.max_level;
	BH_iter<<<grid_size,block_size>>>(bh,thrust::raw_pointer_cast(&d_test_vec[0]));

	std::vector<int > h_test_vec;
	thrust::copy(d_test_vec.begin(),d_test_vec.end(),h_test_vec.begin());

	for(int i=0;i<3;i++)
	{
		std::cout<<h_test_vec[i]<<std::endl;

	}
    return 0;
}
