#include <iostream>
#include "BucketHeap.h"
#include "MinHeap.h"
#include "BucketSignal.h"
#include <assert.h>
#include <math.h>

#include<list>

using namespace std;

int testMinHeap();
int testBucketSignal();
int testBucketHeap();

int main(){
    testMinHeap();
    testBucketSignal();
    testBucketHeap();
    return 0;
}

int testMinHeap(){
    MinHeap* m = new MinHeap(16);
    assert(m->isEmpty());
    m->insertKey(100, 0);
    m->insertKey(200, 1);
    m->insertKey(300, 4);
    m->insertKey(600, 5);
    m->insertKey(700, 6);
    assert(m->getMin().key == 100);
    assert(m->extractMin().key == 100);
    assert(m->extractMin().key == 200);
    m->decreaseKey(m->searchKey(600), 0);
    assert(m->extractMin().key == 600);
    assert(m->getMaxPriority() == 6);

    m = new MinHeap(16);
    m->insertKey(30, 11);
    m->insertKey(40, 12);
    m->insertKey(50, 13);
    m->insertKey(60, 14);
    m->insertKey(70, 15);
    m->insertKey(80, 16);

    int initialSize = m->getSize();
    assert(m->getIthSmallestPriority(3) == 13);
    assert(m->getIthSmallestPriority(4) == 14);
    assert(m->getIthSmallestPriority(1) == 11);
    assert(initialSize == m->getSize());

    int priority = m->getIthSmallestPriority(4);
    Signals* signals = new Signals(16);
    m->moveGreaterPriorityTo(signals, priority);
    assert(signals->getSize() == 2);
    assert(signals->getSignal(0).getKey() == 70);
    assert(signals->getSignal(1).getKey() == 80);
    assert(m->getSize() == initialSize - 2);

    // items to move with equal priority
    m = new MinHeap(16);
    m->insertKey(30, 11);
    m->insertKey(40, 12);
    m->insertKey(50, 13);
    m->insertKey(60, 13);
    m->insertKey(70, 15);
    m->insertKey(80, 16);

    priority = m->getIthSmallestPriority(3);
    assert(priority == 13);
    signals = new Signals(16);
    initialSize = m->getSize();

    m->moveGreaterPriorityTo(signals, priority);
    assert(signals->getSize() == 2);
    assert(m->getSize() == initialSize - 2);
    m->moveSimilarPriorityTo(signals, priority, 1);
    assert(signals->getSize() == 3);
    assert(m->getSize() == initialSize - 3);


//    assert(signals->getSize() != 2);
//    assert(signals->getSignal(1).getKey() != 70);

    int i = 1;
    MinHeap* Bi = new MinHeap(pow(2, (2*i+1)));
    MinHeap* BiPlus1 = new MinHeap(pow(2, (2*(i+1)+1)));

    Bi->insertKey(20, 9);
    Bi->insertKey(20, 5);
    assert((Bi->getIthSmallestPriority(pow(2, 2*i) - Bi->getSize())) == 9);

    BiPlus1->insertKey(20, 19);
    BiPlus1->insertKey(20, 15);
    BiPlus1->insertKey(20, 12);
    BiPlus1->insertKey(20, 11);
    int ithPriority = BiPlus1->getIthSmallestPriority(pow(2, 2*i) - Bi->getSize());
    assert((ithPriority) == 12);

    Bi->moveSmallerPriorityFromBucket(BiPlus1, ithPriority, i);
    assert(Bi->getSize() == 3);
    assert(Bi->getItem(2).priority == 11);
    assert(BiPlus1->getSize() == 3);

    Bi->moveSimilarPriorityFromBucket(BiPlus1, ithPriority, i);
    assert(Bi->getSize() == 4);
    assert(Bi->getItem(3).priority == 12);
    assert(BiPlus1->getSize() == 2);

}

int testBucketSignal(){
    BucketSignal* bs = new BucketSignal(1);
    int i = bs->bucket->getCapacity();
    assert(bs->bucket->getCapacity() == 8); // pow(2, (2*1+1))
    assert(bs->signals->getCapacity() == 16); // pow(2, (2*(1+1)))
    bs = new BucketSignal(4);
    assert(bs->bucket->getCapacity() == pow(2, (2*4+1)));
    assert(bs->signals->getCapacity() == pow(2, (2*(4+1))));

    // delete HandledSignals
    bs->signals->insertKey(UPDATE, 3, 4);
    bs->signals->insertKey(UPDATE, 5, 6);
    bs->signals->insertKey(UPDATE, 15, 16);
    bs->signals->insertKey(UPDATE, 25, 26);
    bs->signals->insertKey(UPDATE, 23, 24);
    bs->signals->insertKey(UPDATE, 35, 36);
    bs->signals->insertKey(UPDATE, 45, 56);
    bs->signals->insertKey(UPDATE, 55, 46);

    assert(bs->signals->getSize() == 8);
    bs->signals->deleteHandledSignals();
    assert(bs->signals->getSize() == 8);
    assert(bs->signals->getSignal(7).getKey() == 55);

    bs->signals->setHandled(5, true);
    bs->signals->setHandled(3, true);
    bs->signals->deleteHandledSignals();
    assert(bs->signals->getSize() == 6);
    assert(bs->signals->getSignal(5).getKey() == 55);
    assert(bs->signals->getSignal(3).getKey() == 23);

}

int testUpdateSignals(){
    BucketHeap* bh = new BucketHeap();
    // add new items
    bh->update(12, 30);
    bh->update(13, 20);
    assert(bh->getIthBucket(1)->getSize() == 2);
    // replace the keys priority
    bh->update(12, 0);
    assert(bh->getIthBucket(1)->getItem(0).key == 12);

}


int testBucketHeap(){
    BucketHeap* bh = new BucketHeap();
    Signals* s = bh->getIthSignal(1);
    assert(s->getCapacity() == 4); // pow(2, (2*1))

    //    testUpdateSignals();

    assert(bh->isEmpty());

    bh->update(3, 19);
    bh->update(4, 20);
    bh->update(5, 29);
    bh->update(6, 39);
    bh->update(7, 50);
    bh->update(12, 110);
    bh->update(13, 119);
    bh->update(14, 120);
    bh->update(15, 129);
    bh->update(16, 139);
    bh->update(17, 150);
    bh->update(2, 10);
    bh->update(16, 1);
    bh->update(17, 1);
    bh->update(4, 200);

    assert(!bh->isEmpty());

    bh->getIthSignal(1)->printItems();
    bh->getIthSignal(2)->printItems();

    cout << bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;
    cout << bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;
    cout << bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;
    cout<<bh->deleteMin().key <<endl;

    assert(bh->isEmpty());

}
