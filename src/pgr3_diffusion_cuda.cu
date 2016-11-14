#include <stdlib.h> 
#include <stdio.h> 
#define BLOCK_SIZE 512 
#define INITIAL_STEPS_SIZE 1024*1024 //assuming everything fits in 1GB of memory
#define DELTA_X 64*1024 //Saying the 1 meter bar is going to be divided by 1 million slices

__device__ double **t;
__device__ long long step = 0;
__device__ double K_d;
__device__ double room_temp_d;
// Per-block pointer 
__global__ void init_global_memory(double K, double room_temp) 
{ 
  
  // Only the first block and the first thread in the block does the allocation 
  // similar to cuda_malloc but for two dimensions structure. 
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    K_d = K;
    room_temp_d = room_temp;
    t = (double**)malloc(INITIAL_STEPS_SIZE * sizeof(double*)); 
    if (t == NULL){
      return;
    }
    t[step] = (double*)malloc(DELTA_X * sizeof(double));
    if(t[step] == NULL){
      return;
    }
  } 
}

__global__ void init_t0()
{  
 
  long long blocks = ceil(DELTA_X / ((float) BLOCK_SIZE)); 
  double *tmpT = t[step];
  if (tmpT != NULL){
    int n = blockDim.x;

    int nTotalThreads;
    if (!n){
      nTotalThreads = n;
    }else{
      //(0 == 2^0)
      int x = 1;
      while(x < n)
      {
        x <<= 1;
      }
      nTotalThreads = x;  
    }

    long long i = blockIdx.x*(nTotalThreads) + threadIdx.x;
    // Only the first thread in the block does the allocation 
    // since we want only one allocation per block. 
    if(i < DELTA_X){
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        tmpT[i] = K_d;
        //printf("First T[%lld] = %f\n", i, tmpT[i]);
      } else {
        tmpT[i] = room_temp_d;
      }
    }
    
    //if(blockIdx.x == blocks-1 && threadIdx.x == nTotalThreads-1){
    //  printf("Last T[%lld] = %f\n", i, tmpT[i]);
    //}
  }
} 

// Increase the number of calcs before convergencer 
__global__ void increase_step(int *result)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    step++;
    atomicAdd(result,1);
    if(step == INITIAL_STEPS_SIZE){
      printf("No more initial memory, TODO: reallocing %lld\n", step);
      return;
    }
    t[step] = (double*)malloc(DELTA_X * sizeof(double));
    if(t[step] == NULL){
      return;
    }
    //printf("Step %lld\n", step);
  }
}

__global__ void calc_diffuse(int *result)
{
  //extern __shared__ bool is_ready_to_stop[];
  if (step == INITIAL_STEPS_SIZE){
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      atomicAdd(result,1);
    }
    return;
  }else{
    double *prev_t = t[step-1];
    double *current_t = t[step];
    if (prev_t != NULL && current_t != NULL){
      int n = blockDim.x;
      long long blocks = ceil(DELTA_X / ((float) BLOCK_SIZE));
      int nTotalThreads;
      if (!n){
        nTotalThreads = n;
      }else{
        int x = 1;
        while(x < n)
        {
          x <<= 1;
        }
        nTotalThreads = x;
      }

      long long i = blockIdx.x*(nTotalThreads) + threadIdx.x;

      double prev;
      double next;

      if(i-1 < 0){
        prev = K_d;
      }else{
        prev = prev_t[i-1];
      }
      if(i+1 >= DELTA_X){
        next = room_temp_d;
      }else{
        next = prev_t[i+1];
      }

      if(i < DELTA_X){
        current_t[i] = (next + prev)/2;
      }
    
      int stop = 0;
      if(current_t[i] == prev_t[i]){
        stop = 1;
      }
      atomicAnd(result, stop);
      /*__syncthreads();
      if(blockIdx.x == 0 && threadIdx.x < 5){
        printf("current_t[%lld] = %f\n", i, current_t[i]);
      }
      if(blockIdx.x == blocks-1 && threadIdx.x > nTotalThreads-5){
        printf("current_t[%lld] = %f\n", i, current_t[i]);
      }
      */
    }
  }
}

// Free Temperature global memory 
__global__ void freemem() 
{ 
  if (blockIdx.x == 0 && threadIdx.x == 0) 
    free(t); 
} 

int main() 
{ 
  int stop = 0;
  int *result_d;
  cudaError_t op_result;
  op_result = cudaMalloc ((void**) &result_d, sizeof(int));
  if (op_result != cudaSuccess) {
    fprintf(stderr, "cudaMalloc (result) failed.");
    exit(1);
  }
  op_result = cudaMemcpy (result_d,&stop, sizeof(int), cudaMemcpyHostToDevice);
  if (op_result != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy host->dev (result) failed.");
    exit(1);
  }
  long long max_heap_size_d = (long long)4*1024*1024*1024;
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_heap_size_d); 
 
  // set execution configuration
  long long block_size = BLOCK_SIZE;
  long long blocks = ceil(DELTA_X / ((float) block_size));
  dim3 dimblock (block_size);
  dim3 dimgrid (blocks);
  // Allocate memory 
  init_global_memory<<< 1, 1 >>>(100.0, 23.0);
  printf("Global Memory Allocated\n");
  init_t0<<< dimgrid, dimblock >>>();	
  printf("T[0] initialized\n");
 
  //int smemSize = dimblock.x * sizeof(bool);
  //cicle
  while(!stop){
    increase_step<<< 1, 1 >>>(result_d);  
    calc_diffuse<<< dimgrid, dimblock >>>(result_d);
    op_result = cudaMemcpy (&stop, result_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (op_result != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy host <- dev (result) failed.");
      exit(1);
    }
  }
  // Free memory 
  freemem<<< 1,1 >>>(); 
  cudaDeviceSynchronize(); 
  return 0; 
}


