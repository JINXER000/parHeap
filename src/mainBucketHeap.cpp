#include <iostream>
#include <assert.h>
#include <list>
#include <chrono>
#include <fstream>


#include "utils.h"
#include "bucketedqueue.h"
#include "parDjikstra.h"
using namespace std;
using namespace std::chrono;



#define USE_GPU


int main(){

	ifstream inputFile;
	ofstream outputFile;
	string inputFileName;
	bool nonDirectedGraph = false;

	int startVertex = 56, destination = 340;
	int numVertices,numEdges;
		int total_rounds=0;

	inputFileName = "input/NetworkScienceGiantComponent.txt";
	openFileToAccess< std::ifstream >( inputFile, inputFileName );
	if( !inputFile.is_open()) {
		std::cerr << "input file not found " << std::endl;
		throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
	}




	//     if the first line of input file specifies num of vertices and edges
	std::string line;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	char* pch;
	std::getline( inputFile, line );
	char cstrLine[256];
	std::strcpy( cstrLine, line.c_str() );

	pch = strtok(cstrLine, delim);
	numVertices = atoi( pch );
	pch = strtok( NULL, delim );
	numEdges = atoi( pch );

	list<struct AdjacentNode>* adjList = new list<struct AdjacentNode>[numVertices];

	int s,v,w;
	for (int i = 0 ; i < numEdges ; i++){
		// cin >> s >> v >> w;

		if(!std::getline( inputFile, line ))
			break;
		std::strcpy( cstrLine, line.c_str() );

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			s = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			v = atoi( pch );
		else
			continue;
		pch=strtok( NULL, delim );
		if( pch != NULL )
			w = atoi( pch );
		adjList[s].push_back({.terminalVertex=v, .weight=w});
	}


	//  int* distance = new int[numVertices];
	std::vector<int> distance(numVertices);
#ifndef USE_GPU
	BucketHeap* bucketHeap = new BucketHeap();
	auto start = high_resolution_clock::now();

	BucketPrioQueue<int> open;
	for (int i = 0 ; i < numVertices ; i++) {
		if (i == startVertex){
			open.push(0,i);
			distance[i] = 0;
		} else {
			open.push(INT_MAX-1,i);
			distance[i] = INT_MAX-1;
		}
		total_rounds++;
	}
	while (!open.empty()) {
		int cur_key = open.pop();
		total_rounds++;
		if (cur_key == destination) break;
		for (struct AdjacentNode n : adjList[cur_key]) {
			if (distance[n.terminalVertex] > distance[cur_key] + n.weight) {
				distance[n.terminalVertex] = distance[cur_key] + n.weight;
				open.push(distance[n.terminalVertex],n.terminalVertex);
			}
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << duration.count() << endl;
	cout << distance[destination] << endl;
	std::cout<<"total rounds= "<<total_rounds<<std::endl;
#else


	Graph<AdjacentNode> cuGraph;
	cuGraph.numEdges=numEdges;
	cuGraph.numVertices=numVertices;
	for (int i = 0; i < numVertices; i++) {
		cuGraph.edgesOffset.push_back(cuGraph.adjacencyList.size());
		cuGraph.edgesSize.push_back(adjList[i].size());
		for (auto &edge: adjList[i]) {
			cuGraph.adjacencyList.push_back(edge);
		}
	}
	// GPU test
	std::vector<int> srcNode;
	for (int i = 0 ; i < numVertices ; i++) {
		if (i == startVertex){
			srcNode.push_back(i);
			distance[i] = 0;
		}else
		{
			distance[i] = INT_MAX-1;
		}
	}

	auto start = high_resolution_clock::now();
	parheap::parDijkstra(srcNode,cuGraph,distance,destination);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << duration.count() << std::endl;

#endif
	return 0;
}
