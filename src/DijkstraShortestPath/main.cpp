#include <iostream>
#include <assert.h>
#include <list>
#include <chrono>

#include "MinHeap.h"

using namespace std;
using namespace std::chrono;


struct AdjacentNode {
    int terminalVertex;
    int weight;
};

int main(){
    struct Vertex currentVertex;
    int temp;

    int startVertex = 0;
    int numVertices, numEdges;
    int s, v, w;
    string line;

    cin >> numVertices >> numEdges;

    MinHeap minHeap(numVertices);
    int* distance = new int[numVertices];

    list<struct AdjacentNode>* adjList = new list<struct AdjacentNode>[numVertices];

    for (int i = 0 ; i < numEdges ; i++){
        cin >> s >> v >> w;
        adjList[s].push_back({.terminalVertex=v, .weight=w});
    }

    auto start = high_resolution_clock::now();
    for (int i = 0 ; i < numVertices ; i++) {
        if (i == startVertex){
            minHeap.insertKey(i, 0);
            distance[i] = 0;
        } else {
            minHeap.insertKey(i, INT_MAX-1);
            distance[i] = INT_MAX-1;
        }
    }

    while (!minHeap.isEmpty()) {
        currentVertex = minHeap.extractMin();
        for (struct AdjacentNode n : adjList[currentVertex.index]) {
            if (distance[n.terminalVertex] > currentVertex.distance + n.weight) {
                distance[n.terminalVertex] = currentVertex.distance + n.weight;
                temp = minHeap.searchKey(n.terminalVertex);
                minHeap.decreaseKey(temp, distance[n.terminalVertex]);
            }
        }
    }
    cout << distance[4] << endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() << endl;

    return 0;
}
