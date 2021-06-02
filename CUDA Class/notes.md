# 1.  Cuda Basics

## Function/Kernel Declaration
- `__global__ void myKernel(argument pointers) {}`
- `__global__` indicates something that runs on the device and is called from
  the host

## Kernel Launch Syntax
- `myKernel<<<N,T>>> (args)`
- the N,1 is the kernel launch configuration
- N = number of blocks. Indicates how many blocks to launch
- T = Number of threads,
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