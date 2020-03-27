#include "BucketHeap.h"

BucketHeap::BucketHeap() {
    signal1 = new Signals(pow(2, (2*1)));
    q = 1;
    BucketSignal* temp = new BucketSignal(1);
    bucketSignals.push_back(*temp);
}

void BucketHeap::update(int x, int p) {
    signal1->insertKey(UPDATE, x, p);
    empty(1);
}

void BucketHeap::deleteItem(int x) {
    signal1->insertKey(DELETE, x, BOGUS_PRIORITY);
    empty(1);
}

struct BucketItem BucketHeap::deleteMin() {
    fill(1);
    MinHeap* B1 = getIthBucket(1);
    assert(B1->getSize()!=0);                   // no items in bucket heap
    return B1->extractMin();
}

void BucketHeap::empty(int i) {
    if (i == q + 1){
        q = q + 1;
//        create two new array Bq and Sq+1
        BucketSignal temp(q);
        bucketSignals.push_back(temp);
    }

//    scan Bi to determine max priority p_dash
    int max_priority = getMaxPriorityOnBucket(i, q);

    // scan Si and Bi and perform signal operations
    MinHeap* Bi = getIthBucket(i);
    Signals* Si = getIthSignal(i);

    Signal tempSignal;
    BucketItem tempBucket;

    int tempBucketIndex;
    for (int j = 0 ; j < Si->getSize(); j++){
        tempSignal = Si->getSignal(j);

        if (tempSignal.getSignalType() == UPDATE){ //update
            tempBucketIndex = Bi->searchKey(tempSignal.getKey());

            if (tempBucketIndex != -1 ){
                // if item is in Bi replace with min priority
                tempBucket = Bi->getItem(tempBucketIndex);
                if (tempSignal.getPriority() < tempBucket.priority) {
                    Bi->decreaseKey(tempBucketIndex, tempSignal.getPriority());
                }
                Si->setHandled(j, true);
            } else {
                // if not in Bi and priority is less than max_priority insert it in Bi
                if (tempSignal.getPriority()<=max_priority) {
                    Bi->insertKey(tempSignal.getKey(), tempSignal.getPriority());
                    //replace tempSignal with delete
                    Si->setSignalType(j, DELETE);
                } else {
                    //do nothing
                }
            }
        } else if (tempSignal.getSignalType() == PUSH) { // push
            tempBucketIndex = Bi->searchKey(tempSignal.getKey());
            if (tempBucketIndex != -1) {
                // if in bucket update priority
                Bi->decreaseKey(tempBucketIndex, tempSignal.getPriority());
            } else {
                // if not in bucket insert
                Bi->insertKey(tempSignal.getKey(), tempSignal.getPriority());
            }
            Si->setHandled(j, true);

        } else if (tempSignal.getSignalType() == DELETE){ // delete
            tempBucketIndex = Bi->searchKey(tempSignal.getKey());
            if (tempBucketIndex != -1) {
                Bi->deleteKey(tempBucketIndex);
            }
        }
    }

    // step 4
    Signals* SiPlus1 = getIthSignal(i+1);
    if (i < q || ! SiPlus1->isEmpty()){//if this is not last bucket or Si+1 has signals
        Signal tempSignal;
        for (int j = 0 ; j < Si->getSize(); j++){
            tempSignal = Si->getSignal(j);
            if (! tempSignal.getHandled()) {
                SiPlus1->insertKey(tempSignal.getSignalType(), tempSignal.getKey(), tempSignal.getPriority());
            }
        }
    }
    Si->reset();

    // step 5
    if (Bi->getSize() > pow(2,  (2 * i))){
        int twoIthSmallestPriority = Bi->getIthSmallestPriority(pow(2, (2 * i)));
        Bi->moveGreaterPriorityTo(SiPlus1, twoIthSmallestPriority);

        int itemsToMove = Bi->getSize() - pow(2, (2 * i));
        Bi->moveSimilarPriorityTo(SiPlus1, twoIthSmallestPriority, itemsToMove);
    }

    // step 6
    if (SiPlus1->getSize() > pow(2, (2*i +1))) {
        empty(i+1);
    }

}

void BucketHeap::fill(int i){
    empty(i);
    Signals* SiPlus1 = getIthSignal(i+1);
    if (!SiPlus1->isEmpty()) {
        empty(i+1);
    }

    MinHeap* BiPlus1 = getIthBucket(i+1);
    MinHeap* Bi = getIthBucket(i);

    int BiPlus1Size = 0;
    if (BiPlus1 != nullptr){
        BiPlus1Size = BiPlus1->getSize();
    }

    if (BiPlus1Size < 2^(2*i) && i < q) {
        fill(i+1);
    }

    // step 3
    if (BiPlus1Size > 0){
        int two_BiIthSmallestPriority = BiPlus1->getIthSmallestPriority(pow(2, (2 * i)) - Bi->getSize());
        Bi->moveSmallerPriorityFromBucket(BiPlus1, two_BiIthSmallestPriority, i);
        Bi->moveSimilarPriorityFromBucket(BiPlus1, two_BiIthSmallestPriority, i);
    }

    // step 4
    q = getNonEmptyBucketSignalIndex();
    maintainNumBuckets();
}

MinHeap* BucketHeap::getIthBucket(int i){
    assert(i <= q + 1);
    int counter=1;
    for (BucketSignal b : bucketSignals){
        if (i == counter){
            return b.bucket;
        }
        counter++;
    }
    return nullptr;
}

Signals* BucketHeap::getIthSignal(int i) {
    if (i == 1) {
        return signal1;
    }
    int counter = 2;
    for (BucketSignal b : bucketSignals){
        if (i == counter){
            return b.signals;
        }
        counter++;
    }
}

int BucketHeap::getMaxPriorityOnBucket(int i, int q) {
    MinHeap* Bi = getIthBucket(i);
    Signals* SiPlus1 = getIthSignal(i+1);

    if (i == q && SiPlus1->isEmpty()){
        return INT_MAX; // +infinity
    }
    if (Bi->isEmpty()){
        return INT_MIN; // -infinity
    }
    return Bi->getMaxPriority();
}

int BucketHeap::getNonEmptyBucketSignalIndex(){
    int maxIndex = 0;
    int counter = 1;
    for (BucketSignal b : bucketSignals){
        if (!b.bucket->isEmpty() || !b.signals->isEmpty()){
            maxIndex = counter;
        }
        counter++;
    }
    return maxIndex;
}

void BucketHeap::maintainNumBuckets() {
    while(q > bucketSignals.size()){
        bucketSignals.pop_back();//? q--?
    }
}