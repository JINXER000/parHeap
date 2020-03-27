#include "MinHeap.h"
#include<climits>
void swap(struct BucketItem *x, struct BucketItem *y);

MinHeap::MinHeap(int cap) {
    heap_size = 0;
    capacity = cap;
    harr = new struct BucketItem[cap];
}

void MinHeap::insertKey(int key, int priority) {
    if (heap_size == capacity) {
//        cout << "\nOverflow: Could not insertKey\n";
        return;
    }

    // First insert the new key at the end
    heap_size++;
    int i = heap_size - 1;

    harr[i].priority = priority;
    harr[i].key = key;

    // Fix the min heap property if it is violated
    while (i != 0 && harr[parent(i)].priority > harr[i].priority) {
        swap(&harr[i], &harr[parent(i)]);
        i = parent(i);
    }
}

///// you cannot decrease key at index 0 !!!!!!!!!
void MinHeap::decreaseKey(int i, int new_val) {

    harr[i].priority = new_val;
    while (i != 0 && harr[parent(i)].priority > harr[i].priority) {
        swap(&harr[i], &harr[parent(i)]);
        i = parent(i);
    }
}

struct BucketItem MinHeap::extractMin() {
    if (heap_size <= 0) {
        struct BucketItem temp;
        return temp;
    }
    if (heap_size == 1) {
        heap_size--;
        return harr[0];
    }

    struct BucketItem root = harr[0];
    harr[0].priority = harr[heap_size-1].priority;
    harr[0].key = harr[heap_size-1].key;
    heap_size--;
    MinHeapify(0);

    return root;
}

void MinHeap::deleteKey(int i) {
    decreaseKey(i, INT_MIN);
    extractMin();
}

void MinHeap::MinHeapify(int i) {
    int l = left(i);
    int r = right(i);
    int smallest = i;
    if (l < heap_size && harr[l].priority < harr[i].priority)
        smallest = l;
    if (r < heap_size && harr[r].priority < harr[smallest].priority)
        smallest = r;
    if (smallest != i) {
        swap(&harr[i], &harr[smallest]);
        MinHeapify(smallest);
    }
}

int MinHeap::searchKey(int keyRef) {
    for (int i = 0; i < heap_size ; i++ ){
        if (harr[i].key == keyRef){
            return i;
        }
    }
    return -1;
}

int MinHeap::getMaxPriority() {
    int maxPriority=INT_MIN;
    for (int i = 0; i < heap_size ; i++ ){
        if (harr[i].priority > maxPriority){
            maxPriority = harr[i].priority;
        }
    }
    return maxPriority;
}

int MinHeap::getIthSmallestPriority(int i){

    MinHeap mh(getSize());
    for (int j = 0; j < getSize(); j++){
        mh.insertKey(harr[j].key, harr[j].priority);
    }

    int priority;
    for (int j = 1; j <= i; j++){
        if (mh.isEmpty()) break;
        priority = mh.extractMin().priority;
    }
    return priority;
}

int MinHeap::moveGreaterPriorityTo(Signals *SiPlus1, int priority) {
    int totalItems = getSize();
    for (int j = 0; j < totalItems; j++){
        if (harr[j].priority > priority){
            SiPlus1->insertKey(PUSH, harr[j].key, harr[j].priority);
            deleteKey(j);
        }
    }
}

int MinHeap::moveSimilarPriorityTo(Signals *SiPlus1, int priority, int itemsToMove) {
    int totalItems = getSize();
    for (int j = 0; j < totalItems && itemsToMove > 0 ; j++){
        if (harr[j].priority == priority){
            SiPlus1->insertKey(PUSH, harr[j].key, harr[j].priority);
            deleteKey(j);
            itemsToMove--;
        }
    }
}

int MinHeap::moveSmallerPriorityFromBucket(MinHeap *BiPlus1, int priority, int i) {
    int totalItems = BiPlus1->getSize();
    for (int j = 0; j < totalItems; j++){
        if (getSize() == pow(2, 2*i) or BiPlus1->isEmpty()){
            break;
        }

        if (BiPlus1->getItem(j).priority < priority){
            insertKey(BiPlus1->getItem(j).key, BiPlus1->getItem(j).priority);
            BiPlus1->deleteKey(j);
        }
    }
}

int MinHeap::moveSimilarPriorityFromBucket(MinHeap *BiPlus1, int priority, int i) {
    int totalItems = BiPlus1->getSize();
    for (int j = 0; j < totalItems; j++){
        if (getSize() == pow(2, 2*i)){
            break;
        }
        assert(getSize() < pow(2, 2*i));

        if (BiPlus1->getItem(j).priority == priority){
            insertKey(BiPlus1->getItem(j).key, BiPlus1->getItem(j).priority);
            BiPlus1->deleteKey(j);
        }
    }
}


void swap(struct BucketItem *x, struct BucketItem *y) {
    int tempDistance = x->priority, tempIndex = x->key;
    x->priority = y->priority;
    x->key = y->key;

    y->priority = tempDistance;
    y->key = tempIndex;
}


void MinHeap::printItems(){
    for (int j = 0; j < getSize(); j++){
        cout<<"("<<harr[j].priority <<"," << harr[j].key<<") ";
    }
    cout<<endl;
}