#ifndef MINHEAP_H
#define MINHEAP_H

#include <iostream>
using namespace std;
#include <assert.h>
#include <math.h>

/**
 * Signal implementation
 */

#define UPDATE 1
#define PUSH 2
#define DELETE 3
#define BOGUS_PRIORITY -46

class Signal {
public:
    int signalType;
    int key;
    int priority;
    bool handled=false;
public:
    Signal() {

    }
    Signal(int sigType, int key_, int priority_)
    {
        signalType = sigType;
        priority = priority_;
        key = key_;

    }
    int getSignalType() const { return signalType; }
    int getKey() { return key; }
    int getPriority() const { return priority; }
    bool getHandled() { return handled; }

    void setHandled(bool handled_) {handled = handled_;}
    void setSignalType(int sigType){signalType = sigType;}
};
class Signals {
public:
    Signal *signals;
    int capacity;
    int size;

    int timestamp;

public:
    Signals(int capacity_) {
        signals = new Signal[capacity_];
        capacity = capacity_;
        size = 0;
    }

    void insertKey(int signalType, int key, int priority){
        if (size+1 > capacity){
            assert(false);
        }
        signals[size] = Signal(signalType, key, priority);
        size += 1;
    }

    void reset() {
        signals = new Signal[capacity];
        size = 0;
    }

    void deleteHandledSignals(){
        int baseIndex = 0;
        for (int i = 0; i < getSize(); i++){
            if (!signals[i].getHandled()){
                signals[baseIndex] = signals[i];
                baseIndex++;
            }
        }
        size = baseIndex;
    }

    bool isEmpty(){ return size == 0;}
    int getSize() {return size;}
    int getCapacity() {return capacity;}
    Signal getSignal(int index){ return signals[index];}
    void setHandled(int index, bool handled){signals[index].setHandled(handled);}
    void setSignalType(int index, int signalType){signals[index].setSignalType(signalType);}

    // DEBUG
    void printItems(){
        cout << "signals :: ";
        for (int i = 0; i < getSize(); i++){
            cout << "(" << signals[i].getSignalType() << ", " << signals[i].getKey() << ", " << signals[i].getPriority() << ") ";
        }
        cout << endl;
    }
};

/**
 * End Signal
 */

struct BucketItem {
    int key;
    int priority;
};

class MinHeap {
public:
    struct BucketItem *harr; // pointer to array of elements in heap
    int capacity; // maximum possible size of min heap
    int heap_size; // Current number of elements in heap
public:
    MinHeap(int capacity);

    void MinHeapify(int);

    int parent(int i) { return (i - 1) / 2; }

    int left(int i) { return (2 * i + 1); }

    int right(int i) { return (2 * i + 2); }

    struct BucketItem extractMin();

    void decreaseKey(int key, int new_val);

    struct BucketItem getMin() { return harr[0]; }

    bool isEmpty() { return heap_size == 0; }

    void deleteKey(int i);

    void insertKey(int key, int priority);

    int searchKey(int keyRef);

    struct BucketItem getItem(int index) {return harr[index];}

    int getSize(){return heap_size;}
    int getCapacity(){return capacity;}
    int getMaxPriority();
    int getIthSmallestPriority(int i);

    int moveGreaterPriorityTo(Signals* SiPlus1, int priority);
    int moveSimilarPriorityTo(Signals* SiPlus1, int priority, int itemsToMove);

    int moveSmallerPriorityFromBucket(MinHeap *BiPlus1, int priority, int i);
    int moveSimilarPriorityFromBucket(MinHeap *BiPlus1, int priority, int i);

    // DEBUG
    void printItems();

};


#endif //_MINHEAP_H
