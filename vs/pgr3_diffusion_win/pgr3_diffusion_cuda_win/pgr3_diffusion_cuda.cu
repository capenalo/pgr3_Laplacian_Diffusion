#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h> 
#include <stdio.h> 

#define THREADS_PER_BLOCK 32 //1024 total threads per block

#define DELTA_X 1024 //This is a square of 1024
#define DELTA_Y 1024 //    by 1024

#define MAX_STEPS_KERNEL 10 //number of computations in kernel per cycle
#define MIN_VARIATION 0.00005

__device__ int step = 0;

__global__ void calc_diffuse(double *current_mat, double *prev_mat, size_t rows, size_t cols, int *partial_result)
{
	if (step == MAX_STEPS_KERNEL){
		if (blockIdx.x == 0 && threadIdx.x == 0) {
			*partial_result = 1;
		}
		return;
	} else {
		int col = blockIdx.x*blockDim.x + threadIdx.x;
		int row = blockIdx.y*blockDim.y + threadIdx.y;

		if ((row < rows) && (col < cols)) {
			if (row == 0 && col == 0) {
				current_mat[(row)*cols + (col)] = (100.0 + prev_mat[(row + 1)*cols + (row)] + 100 + prev_mat[(row)*cols + (col + 1)]) / 4;
			}
			else if (row == 0 && col <= cols - 1) {
				current_mat[(row)*cols + (col)] = (23 + prev_mat[(row + 1)*cols + (row)] + prev_mat[(row)*cols + (col - 1)] + prev_mat[(row)*cols + (col + 1)]) / 4;
			}
			else if (row == 0 && col == cols - 1) {
				current_mat[(row)*cols + (col)] = (23 + prev_mat[(row + 1)*cols + (row)] + prev_mat[(row)*cols + (col - 1)] + 23) / 4;
			}
			else if (row < rows -1 && col == 0) {
				current_mat[(row)*cols + (col)] = (prev_mat[(row - 1)*cols + (col)] + prev_mat[(row + 1)*cols + (row)] + 23 + prev_mat[(row)*cols + (col + 1)]) / 4;
			}
			else if (row == rows-1 && col == 0) {
				current_mat[(row)*cols + (col)] = (prev_mat[(row - 1)*cols + (col)] + 23 + 23 + prev_mat[(row)*cols + (col + 1)]) / 4;
			}
			else if (row == rows - 1 && col < cols - 1) {
				current_mat[(row)*cols + (col)] = (prev_mat[(row - 1)*cols + (col)] + 23 + prev_mat[(row)*cols + (col - 1)] + prev_mat[(row)*cols + (col + 1)]) / 4;
			}
			else if (row == rows - 1 && col == cols - 1) {
				current_mat[(row)*cols + (col)] = (prev_mat[(row - 1)*cols + (col)] + 23 + prev_mat[(row)*cols + (col - 1)] + 23) / 4;
			}
			else {
				current_mat[(row)*cols + (col)] = (prev_mat[(row - 1)*cols + (col)] + prev_mat[(row + 1)*cols + (row)] + prev_mat[(row)*cols + (col - 1)] + prev_mat[(row)*cols + (col + 1)]) / 4;
			}
			atomicAdd(&step, 1);
			
		}
	}
}
__global__ void debug_step(double *mat, size_t cols) {
	int new_step = step;
	int row = threadIdx.y;
	int col = threadIdx.x;
	double tmp = mat[(row)*cols + (col)];
}

void cudasafe(cudaError_t error, char* message)
{
	if (error != cudaSuccess) {
		fprintf(stderr, "ERROR: %s : %i\n", message, error); exit(-1);
	}

}

int init_t0(double *mat1, size_t rows, size_t cols, double K, double room_temp)
{
	int i;
	int j;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			mat1[i*cols + j] = room_temp;
		}
	}
	mat1[0] = K;
	return 0;
}

void print_matrix(double *mat, size_t rows, size_t cols) {
	int i;
	int j;
	printf("[");
	for (i = 0; i < rows; i++) {
		printf("[");
		for (j = 0; j < cols; j++) {
			printf(" %.5f ",mat[i*cols + j]);
		}
		printf("]\n");
	}
	printf("]\n");
}

int main() 
{ 
	int stop = 0;
	
	double *mat1;
	double *mat2;
	
	mat1 = (double*)calloc(DELTA_X * DELTA_Y,sizeof(double));
	mat2 = (double*)calloc(DELTA_X * DELTA_Y,sizeof(double));
	
	init_t0(mat1, DELTA_X, DELTA_Y, 100.0, 23.0);
	printf("MAT1\n");
	double *mat1_d, *mat2_d;
	int *result_d;

	cudasafe(cudaMalloc ((void**) &result_d, sizeof(int)), "cudaMalloc(result) failed.");
	cudasafe(cudaMemcpy (result_d,&stop, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy host->dev (result) failed.");

	cudasafe(cudaMalloc((void**)&mat1_d, DELTA_X * DELTA_Y * sizeof(double)), "cudaMalloc(mat1_d) failed.");
	cudasafe(cudaMemcpy(mat1_d, mat1, DELTA_X * DELTA_Y * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy host->dev (mat1_d) failed.");

	cudasafe(cudaMalloc((void**)&mat2_d, DELTA_X * DELTA_Y * sizeof(double)), "cudaMalloc(mat2_d) failed.");
	cudasafe(cudaMemcpy(mat2_d, mat2, DELTA_X * DELTA_Y * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy host->dev (mat2_d) failed.");

	long long max_heap_size_d = (long long)3*1024*1024*1024;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, max_heap_size_d); 
 
	// set execution configuration
	long long block_size = THREADS_PER_BLOCK;
	long long blocks_x = ceil(DELTA_X / ((float) block_size));
	long long blocks_y = ceil(DELTA_Y / ((float) block_size));
	dim3 dimblock (THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 dimgrid (blocks_x, blocks_y);

	calc_diffuse <<< dimgrid, dimblock >>>(mat2_d, mat1_d, DELTA_X, DELTA_Y, result_d);
	debug_step << <1, 1 >> > (mat2_d, DELTA_X);
	cudasafe(cudaMemcpy(mat2, mat2_d, DELTA_X * DELTA_Y * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy host <-dev(mat2_d) failed.");
	print_matrix(mat2, DELTA_X, DELTA_Y);
	cudaDeviceSynchronize(); 
	printf("Press any key to exit!");
	getchar();
 
	return 0; 
}
