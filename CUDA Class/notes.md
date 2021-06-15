# 1.  Cuda Basics

## Function/Kernel Declaration
- `__global__ void myKernel(argument pointers) {}`
- `__global__` indicates something that runs on the device and is called from
  the host

## Kernel Launch Syntax
- `myKernel<<<N,T>>> (args)`
- the N,1 is the kernel launch configuration
- N = number of blocks. Indicates how many blocks to launch. Can be of type dim3
- T = Number of threads, can be of type dim3
- Hierarchy
  - Grid: all blocks and threads
  - Blocks, can contain threads
  - Threads
- Typically N and T are found by N = size of array / threads per block, and T =
  threads per block
  - Array length might not divide evenly, for now just add an `if` statement to
    stop extra threads
  - So typically launch looks like `kernel<<< (N + T-1) / T, T>>> (args, N)`
    - where N is the size of the array and T is the number of threads per block

## Indices
- Block index = `blockIdx.x`
  - `blockIdx` has three elements `.x`, `.y`, and `.z`
- Thread index = 'threadIdx.x`
  - `threadIdx` has three elements `.x`, `.y`, and `.z`
  - cannot be more than 1024 threads per block
- Both of these are of type `dim3`. A 3 element structure where each element is
  referenced with `.x`, `.y`, or `.z`.
- Global unique index example
  - `int index = threadIdx.x + blockIdx.x * M` where M is the size of each block
  - `M` is usually found with `blockDim.x` which is the size of the blocks in
    direction x. i.e. the number of threads within a block

## Memory Management
- `cudaMalloc()`, `cudaFree()`, and `cudaMemcpy()`
- Usually if `a` is the host pointer then `d_a` is the device pointer
- syntax:
  - `cudaMalloc((void **) &d_a, size)` where size is the size in bytes, d_a is
    the pointer
  - copy to device `cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice)`
  - copy to host   `cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost)`
    - first arg is destination, second is source, third is size in bytes, fourth
      is direction
  - `cudaFree(d_a)` clean up memory
- `size` can be found with `sizeof(type)` then multiply by the number of that
  type

## Threads vs. Blocks
- Threads can communicate and synchronize within a block but not between blocks

## Misc.
- `cudaDeviceSynchronize()` synchronizes host and device




# 2. Shared Memory

## Stencil Operation
- Use a certain number of grid points on either side of the 'center' point to
  produce a new center point
- Has a width and radius. The radius is the number of points on either side to
  be used. The width is 2x radius + 1 for the center point
- Threads can generally run in any order, so you still need to make sure
  blocks/synchronizes are there to avoid race conditions etc
  - `__syncthreads()` is the block level barrier. Does not operate between Blocks
  - all threads MUST be able to hit the `__syncthreads()`. I.e. you can use it
    in a conditional

## Shared Memory
- shared memory is on-chip memory similar to cache on a CPU
  - it used user managed unlike a cache though
  - Limited to ~48 kB per thread block. I.e. ~6,000 doubles
- ~5 times faster than global memory, i.e. memory on external DRAM chips
- shared between threads in a block but not between blocks
- declaration syntax: `__shared__ int arrName[BLOCK_SIZE + 2 * RADIUS]`
  - called inside of kernels
  - Dynamic (i.e. run time) memory is allocated by omiting the size of the array
    (i.e. just `int arrName[]`) then passing a third argument in the triple
    chevron that is the size in bytes. i.e. `kernelLaunch<<<grid, block,
    shared_size_in_bytes>>>(args)`
  - Might need to add `extern` keyword in front of `__shared__` for dynamic memory
- Now we'll need a global index and an index for shared/local memory
  - `int lindex = threadIdx.x + RADIUS`
- Read in memory
  - Read in the non-halo/ghost pixels `temp[lindex = inputArr[globalIndex]`
  - Read in halo pixels: Do the same thing but only with the first RADIUS
    threads to load in the halo
- Save results in global memory since we won't need it multiple times
- Communication can happen in shared memory via writing with one thread and
  reading with another


# 3-4. Cuda Optimization

## Warps
- a thread block consists of 1 or more warps of 32 threads
- a warp is executed physically in parallel on SM
- all instructions are warp wide and all threads in the warp operate in lockstep
- Threads per block should be a multiple of the warp size

## Launch Configuration
- Instructions are always issued in order. There is no out-of-order-execution
  like with x86
  - This is not true for warps. Individual threads are executed in order but
    warps can be executed out of order
- Memory reads are non-blocking.
- Threads can stall while waiting for data
- latency:
  - Global memory retrieval latency (GMEM) >100 clockcycles, sometimes as high
    as 400
  - Arithmetic latency <100 cycles time between instruction issue and
    instruction completion. Typically 5-10 cycles
- Launch enough threads to hide latency
- Limit of 64 warps per SM (i.e. 2048 threads). This should be the goal.
  Maximize the number of threads. We're shooting for ~160,000 threads per Volta
  processor
  - Note that you are limited to 1024 threads per block so you'll usually want
    to schedule multiple thread blocks per SM. 512 threads per block usually
    works well
  - This is equivalent to roughly a 55x55x55 grid in Cholla which is (I think)
    about 11.25MB worth of grid

## Memory Hierarchy
- Arranged fastest to slowest/closest to processor to farthest
  - Local storage: Registers mostly, managed by compiler
  - Shared Memory/L1 Cache:
    - Shared memory is on chip and can be managed by the programmer. Typically
      48, 64, or 96 KB of shared memory
    - L1 Cache: Compiler managed. Per SM/Per block resource
  - L2 Cache: All accesses go through this, is device wide
  - Constant and Texture Caches: fixme idk what these are. They might be purely
    logical
  - Global Memory: latency in hundreds of cycles, throughput is ~900 GB/s (V100)
- **Writing**
  - Invalidate the updated data in L1 and update the value in L2 to later be
    written to global memory
- **Non-Caching**: Just skip requests to L1. Can be used to communicate between
  blocks by forcing the threads to go to L2. Can sometimes increase performance
  slightly

## Load Operation
- Run warp wide, though each thread can request different memory
- Try to load large amounts of memory per thread
- Loading consecutive memory locations is usually much faster and uses the bus
  more efficiently
- Getting a single value (like gamma) from memory is very bus inefficient and
  occasionally is fine but should be avoided if possible. Try to deal with this
  by maybe loading multiple things like that in a single vector/array
- Requesting random bytes is worst case just like the single value request. This
  could be an issue with MHD/Constrained Transport
  - non-caching loads can improve this. L1 cache requires 128 byte loads but
    global memory only requires 32 byte loads. So on semi-random loads you can
    significantly improve efficiency with non-caching loads
  - Since all the CT fields are used this probably isn't a huge issue

## Shared Memory
- 32 banks of 4 byte sections
- Bank 0 has bytes 0-3 and 128-131
- Think about it like a 2D array. 32 banks wide and X rows high where X is
  either the total size or just enough for your usage
- every bank can read out simultaneously, but only 4 bytes per bank
- Try to access all bank simultaneously, avoid accessing one bank many many
  times at once, can be done with 'padding'. Add an extra 4 bytes to the end of
  each "row", this will offset memory and allow both columnar and row access at
  nearly full speed


# 5. Atomics, Reductions, and Warp Shuffle


## Atomics and Serial Reductions
- Add: `atomicAdd(*pointer to output*, *value to add*)`
  - Allows only 1 thread to use the output location at a time
  - serializes threads
  - Implemented in L2 Cache
  - has poor performance
- **Other Atomics:**
  - `atomicMax/Min`
  - `atomicAdd/Sub`
  - `atomicInc/Dec` increment or decrement value with rollover
  - `atomicExch/CAS` swap or conditionally swap values
  - `atomicAnd/Or/Xor` bitwise operations
  - atomics have different datatypes they work with
  - `atomicCas` can do lots of interesting things
  - [Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
- **Returns**
  - Atomics return the old value in that location
  - can be used to choose work items, queue slots, etc. i.e. to order operations


## Parallel Reductions & Global Synchronization
- Options:
  - Split into different kernels since CUDA executes different kernel calls serially
  - use atomics
  - threadblock draining (`threadFenceReduction`, sample code available)
  - Cooperative Groups - to be covered later
- General Idea:
  1. Perform a grid stride loop to account for varying number of threads/data size
  2. Loop from the highest to lowest level of the reduction with `__syncthreads`
     keeping stuff synced
     1. Do a parallel sweep reduction (i.e. x[i] += x[i+threadNum])
  4. Use atomics to sync between blocks

## Warp Shuffle
- See slide 21
- Direct communication between threads without using shared memory
- Only works *within* a warp
- API:
  - `__shfl_sync()` - Copy from arbitrary lane ID
  - `__shfl_xor_sync()` - Copy from calculated ID
  - `__shfl_up_sync()` - Copy from lower lane
  - `__shfl_down_sync()` - Copy from higher lane
- Both threads must "participate"
  - Can cause issues with `if` statements that stop certain threads from
    executing
- Reduces shared memory usage per thread block. Also reduces shared memory
  pressure
- Warp level broadcast
- warp level sum/atomic aggregation
  - Reduces the number of atomic operation by a factor of 32