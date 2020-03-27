#include <iostream>
#include <assert.h>
#include <list>
#include <chrono>
#include <fstream>


#include "BucketHeap.h"
#include "BucketSignal.h"
#include "utils.h"

#include "parbucket.h"
using namespace std;
using namespace std::chrono;





int main(){

    ifstream inputFile;
    ofstream outputFile;
    string inputFileName;
    bool nonDirectedGraph = false;
    
    struct BucketItem currentVertex;
    int startVertex = 0, destination = 348;
    int numVertices,numEdges;

    inputFileName = "input/NetworkScienceGiantComponent.txt";
    openFileToAccess< std::ifstream >( inputFile, inputFileName );
    if( !inputFile.is_open()) {
        std::cerr << "input file not found " << std::endl;
        throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
    }
    std::vector<initial_vertex> parsedGraph( 0 );
    numEdges = parse_graph(
            inputFile,		// Input file.
            parsedGraph,	// The parsed graph.
            startVertex,
            nonDirectedGraph );		// Arbitrary user-provided parameter.

    numVertices= parsedGraph.size();



    BucketHeap* bucketHeap = new BucketHeap();

    // if the first line of input file specifies num of vertices and edges
    // std::string line;
    // char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	// char* pch;
    // std::getline( inputFile, line );
    // char cstrLine[256];
    // std::strcpy( cstrLine, line.c_str() );

    // pch = strtok(cstrLine, delim);
    // numVertices = atoi( pch );
    // pch = strtok( NULL, delim );
    // numEdges = atoi( pch );


    int* distance = new int[numVertices];
    // list<struct AdjacentNode>* adjList = new list<struct AdjacentNode>[numVertices];

    // int s,v,w;
    // for (int i = 0 ; i < numEdges ; i++){
    //     // cin >> s >> v >> w;

    //     if(!std::getline( inputFile, line ))
    //         break;  
    //     std::strcpy( cstrLine, line.c_str() );

    //     pch = strtok(cstrLine, delim);
	// 	if( pch != NULL )
	// 		s = atoi( pch );
	// 	else
	// 		continue;
	// 	pch = strtok( NULL, delim );
	// 	if( pch != NULL )
	// 		v = atoi( pch );
	// 	else
	// 		continue;
    //     pch=strtok( NULL, delim );
    //     if( pch != NULL )
	// 		w = atoi( pch );
    //     adjList[s].push_back({.terminalVertex=v, .weight=w});
    // }


    

    auto start = high_resolution_clock::now();
    for (int i = 0 ; i < numVertices ; i++) {
        if (i == startVertex){
            bucketHeap->update(i, 0);
            distance[i] = 0;
        } else {
            bucketHeap->update(i, INT_MAX-1);
            distance[i] = INT_MAX-1;
        }
    }

    while (!bucketHeap->isEmpty()) {
        currentVertex = bucketHeap->deleteMin();

        if (currentVertex.key == destination) break;

        // for (struct AdjacentNode n : adjList[currentVertex.key]) {
        //     if (distance[n.terminalVertex] > currentVertex.priority + n.weight) {
        //         distance[n.terminalVertex] = currentVertex.priority + n.weight;
        //         bucketHeap->update(n.terminalVertex, distance[n.terminalVertex]);
        //     }
        // }

        for(struct neighbor n: parsedGraph.at(currentVertex.key).nbrs)
        {
            if(distance[n.dstIndex]>currentVertex.priority+n.weight)
            {
                distance[n.dstIndex]=currentVertex.priority+n.weight;
                bucketHeap->update(n.dstIndex, distance[n.dstIndex]);
            }
        }
    }
    cout << distance[destination] << endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << endl;
    parDijkstra();
    return 0;
}
