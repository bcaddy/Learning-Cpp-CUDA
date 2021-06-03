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