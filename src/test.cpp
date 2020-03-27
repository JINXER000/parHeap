int numVertices = 6;
int startVertex = 0;
MinHeap minHeap(numVertices);
int distance[numVertices];

list<struct AdjacentNode> adjList[numVertices];

adjList[0].push_front({.terminalVertex=1, .weight=4});
adjList[0].push_back({.terminalVertex=3, .weight=2});

adjList[1].push_back({.terminalVertex=0, .weight=4});
adjList[1].push_back({.terminalVertex=2, .weight=3});
adjList[1].push_back({.terminalVertex=4, .weight=3});

adjList[2].push_back({.terminalVertex=1, .weight=3});
adjList[2].push_back({.terminalVertex=5, .weight=2});

adjList[3].push_back({.terminalVertex=0, .weight=2});
adjList[3].push_back({.terminalVertex=4, .weight=3});

adjList[4].push_back({.terminalVertex=3, .weight=3});
adjList[4].push_back({.terminalVertex=1, .weight=3});
adjList[4].push_back({.terminalVertex=5, .weight=1});

adjList[5].push_back({.terminalVertex=4, .weight=1});
adjList[5].push_back({.terminalVertex=2, .weight=2});






int numVertices = 6;
int startVertex = 0;
MinHeap minHeap(numVertices);
int distance[numVertices];

list<struct AdjacentNode> adjList[numVertices-1];

adjList[0].push_front({.terminalVertex=1, .weight=4});
adjList[0].push_back({.terminalVertex=3, .weight=2});

adjList[1].push_back({.terminalVertex=0, .weight=4});
adjList[1].push_back({.terminalVertex=2, .weight=3});
adjList[1].push_back({.terminalVertex=4, .weight=3});

adjList[2].push_back({.terminalVertex=1, .weight=3});
adjList[2].push_back({.terminalVertex=5, .weight=2});

adjList[3].push_back({.terminalVertex=0, .weight=2});
adjList[3].push_back({.terminalVertex=4, .weight=3});

adjList[4].push_back({.terminalVertex=3, .weight=3});
adjList[4].push_back({.terminalVertex=1, .weight=3});
adjList[4].push_back({.terminalVertex=5, .weight=1});

adjList[5].push_back({.terminalVertex=4, .weight=1});
adjList[5].push_back({.terminalVertex=2, .weight=2});