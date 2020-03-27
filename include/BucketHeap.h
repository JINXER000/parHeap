#ifndef BUCKETHEAP_H
#define BUCKETHEAP_H

#include <list>
#include <math.h>
#include <assert.h>

#include "BucketSignal.h"
#include "limits.h"
using namespace std;

class BucketHeap {
    int capacity;
    int num_items;

    int num_buckets;
    Signals* signal1;
    list<BucketSignal> bucketSignals;

    int q;

public:
    BucketHeap();

    void update(int x, int p);
    struct BucketItem deleteMin();
    void deleteItem(int x);


    void empty(int signalIndex);
    void fill(int bucketIndex);

    int getNumBucket() { return num_buckets;}

    MinHeap* getIthBucket(int i);
    Signals* getIthSignal(int i);

    int getMaxPriorityOnBucket(int i, int q);

    int getNonEmptyBucketSignalIndex();

    bool isEmpty(){
        return getIthBucket(1)->isEmpty();
    }
    void maintainNumBuckets();

};


#endif //BUCKETHEAP_H
