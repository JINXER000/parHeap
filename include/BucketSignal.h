#ifndef BUCKETSIGNAL_H
#define BUCKETSIGNAL_H

#include "MinHeap.h"
#include <queue>
#include <array>
#include <assert.h>
#include <math.h>

using namespace std;


/**
 * Bucket Signal implementation
 */
class BucketSignal {

public:
    MinHeap *bucket;
    Signals *signals;
    int bucketIndex;
    int bucketCapacity;
    int signalCapacity;

    BucketSignal(int i){
        bucketIndex = i;
        bucketCapacity = pow(2, (2*i+1));     // Bi
        signalCapacity = pow(2, (2*(i+1))); // Si+1
        bucket = new MinHeap(bucketCapacity);
        signals = new Signals(signalCapacity);
    }
};

#endif //BUCKETSIGNAL_H
