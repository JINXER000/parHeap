/*
 * parDjikstra.h
 *
 *  Created on: Mar 27, 2020
 *      Author: yzchen
 */

#ifndef PARDJIKSTRA_H_
#define PARDJIKSTRA_H_

template <typename Vtype>
struct Graph {
    std::vector<Vtype> adjacencyList; // all edges
    std::vector<int> edgesOffset; // offset to adjacencyList for every vertex
    std::vector<int> edgesSize; //number of edges for every vertex
    int numVertices = 0;
    int numEdges = 0;
};

int parDijkstra(std::vector<int> &srcNode,
		Graph<AdjacentNode> &cuGraph,
		std::vector<int> &distances);



#endif /* PARDJIKSTRA_H_ */
