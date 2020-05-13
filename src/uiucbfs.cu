# include "ubfsfunc.h"
# include "uiucbfs.cuh"

namespace ubfs
{

void parWavefront(std::vector<int> &srcNode,
		Graph<AdjacentNode> &cuGraph,
		std::vector<int> &distances,
		int destination)
{
	thrust::device_vector<Node> d_graph_node(cuGraph.numEdges);
	thrust::device_vector<Edge> d_graph_edge(cuGraph.numVertices);

	for(int i=0;i<cuGraph.numEdges;i++)
	{
		d_graph_node[i].x=cuGraph.adjacencyList[i].terminalVertex;
		d_graph_node[i].y=cuGraph.adjacencyList[i].weight;
	}
	for(int i=0;i<cuGraph.numVertices;i++)
	{
		d_graph_edge[i].x=cuGraph.edgesOffset[i];
		d_graph_edge[i].y=cuGraph.edgesSize[i];
	}
	using thrust::raw_pointer_cast;
	  //bind the texture memory with global memory
	  cudaBindTexture(0,g_graph_node_ref,raw_pointer_cast(d_graph_node), sizeof(Node)*cuGraph.numVertices);
	  cudaBindTexture(0,g_graph_edge_ref,raw_pointer_cast(d_graph_edge), sizeof(Edge)*cuGraph.numEdges);


	 int* d_color;
	  cudaMalloc((void**) &d_color, sizeof(int)*cuGraph.numVertices);
	  int* d_cost;
	  cudaMalloc((void**) &d_cost, sizeof(int)*cuGraph.numVertices);
	  int * d_q1;
	  int * d_q2;
	  cudaMalloc((void**) &d_q1, sizeof(int)*cuGraph.numVertices);
	  cudaMalloc((void**) &d_q2, sizeof(int)*cuGraph.numVertices);
	  int * tail;
	  cudaMalloc((void**) &tail, sizeof(int));
	  int *front_cost_d;
	  cudaMalloc((void**) &front_cost_d, sizeof(int));
	  CUDA_DEV_MEMSET(d_color,WHITE,sizeof(int)*cuGraph.numVertices);
	  CUDA_DEV_MEMSET(d_cost,0,sizeof(int)*cuGraph.numVertices);

	  cudaMemcpy(tail,&h_top,sizeof(int),cudaMemcpyHostToDevice);
	  cudaMemcpy(&d_cost[srcNode[0]],0,sizeof(int),cudaMemcpyHostToDevice);

	  cudaMemcpy( &d_q1[0], &srcNode[0], sizeof(int), cudaMemcpyHostToDevice);



	  //whether or not to adjust "k", see comment on "BFS_kernel_multi_blk_inGPU" for more details
	  int * switch_kd;
	  cudaMalloc((void**) &switch_kd, sizeof(int));
	  int * num_td;//number of threads
	  cudaMalloc((void**) &num_td, sizeof(int));

	  //whether to stay within a kernel, used in "BFS_kernel_multi_blk_inGPU"
	  bool *stay;
	  cudaMalloc( (void**) &stay, sizeof(bool));
	  int switch_k;

	  //max number of frontier nodes assigned to a block
	  int * max_nodes_per_block_d;
	  cudaMalloc( (void**) &max_nodes_per_block_d, sizeof(int));
	  int *global_kt_d;
	  cudaMalloc( (void**) &global_kt_d, sizeof(int));
	  cudaMemcpy(global_kt_d,&zero, sizeof(int),cudaMemcpyHostToDevice);

	  int h_overflow = 0;
	  int *d_overflow;
	  cudaMalloc((void**) &d_overflow, sizeof(int));
	  cudaMemcpy(d_overflow, &h_overflow, sizeof(int), cudaMemcpyHostToDevice);

	  int num_t;//number of threads
	  int k=0;//BFS level index
	  int num_of_blocks;
	  int num_of_threads_per_block;


	  do
	   {
	     cudaMemcpy( &num_t, tail, sizeof(int), cudaMemcpyDeviceToHost);
	     cudaMemcpy(tail,&zero,sizeof(int),cudaMemcpyHostToDevice);

	     if(num_t == 0){//frontier is empty
	       cudaFree(stay);
	       cudaFree(switch_kd);
	       cudaFree(num_td);
	       break;
	     }

	     num_of_blocks = 1;
	     num_of_threads_per_block = num_t;
	     if(num_of_threads_per_block <NUM_BIN)
	       num_of_threads_per_block = NUM_BIN;
	     if(num_t>MAX_THREADS_PER_BLOCK)
	     {
	       num_of_blocks = (int)ceil(num_t/(double)MAX_THREADS_PER_BLOCK);
	       num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	     }
	     if(num_of_blocks == 1)//will call "BFS_in_GPU_kernel"
	       num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	     if(num_of_blocks >1 && num_of_blocks <= NUM_SM)// will call "BFS_kernel_multi_blk_inGPU"
	       num_of_blocks = NUM_SM;

	     //assume "num_of_blocks" can not be very large
	     dim3  grid( num_of_blocks, 1, 1);
	     dim3  threads( num_of_threads_per_block, 1, 1);

	     if(k%2 == 0){
	       if(num_of_blocks == 1){
	         BFS_in_GPU_kernel<<< grid, threads >>>(d_q1,d_q2, raw_pointer_cast(d_graph_node),
	             raw_pointer_cast(d_graph_edge), d_color, d_cost,num_t , tail,GRAY0,k,d_overflow);
	       }
	       else if(num_of_blocks <= NUM_SM){
	         (cudaMemcpy(num_td,&num_t,sizeof(int),
	                     cudaMemcpyHostToDevice));
	         BFS_kernel_multi_blk_inGPU
	           <<< grid, threads >>>(d_q1,d_q2, raw_pointer_cast(d_graph_node),
	               raw_pointer_cast(d_graph_edge), d_color, d_cost, num_td, tail,GRAY0,k,
	               switch_kd, max_nodes_per_block_d, global_kt_d,d_overflow);
	         (cudaMemcpy(&switch_k,switch_kd, sizeof(int),
	                     cudaMemcpyDeviceToHost));
	         if(!switch_k){
	           k--;
	         }
	       }
	       else{
	         BFS_kernel<<< grid, threads >>>(d_q1,d_q2, raw_pointer_cast(d_graph_node),
	             raw_pointer_cast(d_graph_edge), d_color, d_cost, num_t, tail,GRAY0,k,d_overflow);
	       }
	     }
	     else{
	       if(num_of_blocks == 1){
	         BFS_in_GPU_kernel<<< grid, threads >>>(d_q2,d_q1, raw_pointer_cast(d_graph_node),
	             raw_pointer_cast(d_graph_edge), d_color, d_cost, num_t, tail,GRAY1,k,d_overflow);
	       }
	       else if(num_of_blocks <= NUM_SM){
	         (cudaMemcpy(num_td,&num_t,sizeof(int),
	                     cudaMemcpyHostToDevice));
	         BFS_kernel_multi_blk_inGPU
	           <<< grid, threads >>>(d_q2,d_q1, raw_pointer_cast(d_graph_node),
	               raw_pointer_cast(d_graph_edge), d_color, d_cost, num_td, tail,GRAY1,k,
	               switch_kd, max_nodes_per_block_d, global_kt_d,d_overflow);
	         (cudaMemcpy(&switch_k,switch_kd, sizeof(int),
	                     cudaMemcpyDeviceToHost));
	         if(!switch_k){
	           k--;
	         }
	       }
	       else{
	         BFS_kernel<<< grid, threads >>>(d_q2,d_q1, raw_pointer_cast(d_graph_node),
	             raw_pointer_cast(d_graph_edge), d_color, d_cost, num_t, tail, GRAY1,k,d_overflow);
	       }
	     }
	     k++;
	     cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
	     if(h_overflow) {
	       printf("Error: local queue was overflown. Need to increase W_LOCAL_QUEUE\n");
	       return;
	     }
	   } while(1);
	   cudaThreadSynchronize();
	   pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	   printf("GPU kernel done\n");

	   // copy result from device to host
	   cudaMemcpy(h_cost, d_cost, sizeof(int)*cuGraph.numVertices, cudaMemcpyDeviceToHost);
	   cudaMemcpy(color, d_color, sizeof(int)*cuGraph.numVertices, cudaMemcpyDeviceToHost);
	   cudaUnbindTexture(g_graph_node_ref);
	   cudaUnbindTexture(g_graph_edge_ref);

	   cudaFree(d_color);
	   cudaFree(d_cost);
	   cudaFree(tail);
	   cudaFree(front_cost_d);

	   // cleanup memory
	   free( h_graph_nodes);
	   free( h_graph_edges);
	   free( color);

}

}
