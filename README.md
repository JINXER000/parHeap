### note

### dynamic parallel
- in dynamic parallelism, you cannot use recursion in a fuction with shared mem. Either the stack will overflow or the compiler will not permit your compilation. 

- child grid is not necessarily finish until you call device_sync function in parent grid.
See https://www.it1352.com/587905.html




## todo
- is not fully parallelized. Dynamic parallel is slower than serail. May be CUB is faster?

- The DELETE operation in paper is confusing. It will always sit in S0 if setting its priority as -1. So in this repo I just remove it, although it may cause some problem when duplicate keys are in non-adjacent levels--- they may never met. But since it is sorted, it may cause no problem in SSSP.

- may be the atomic lock is slow-- can we just seperate the levels in odd and even?
No I think because , if we seperate , we have to let CPU take charge of it(for syncronization), thus smaller level have to wait for larger level for they take longer time.

So another solution may be, let thread 0 of each block take care of the atomic lock. Then use other threads to perform scan. 

- in wavefront, the first thousands of points can be just filled in without resolve, since their priority are all 0. 


- counter check: how many rounds does cpu and gpu perform?

