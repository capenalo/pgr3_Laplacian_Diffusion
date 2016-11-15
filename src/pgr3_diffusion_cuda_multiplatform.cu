#ifdef OS_WINDOWS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define THREADS_PER_BLOCK 32 //1024 total threads per block
#else
#define THREADS_PER_BLOCK 16 //512 total threads per block
#include <sys/time.h>
#endif

#include <stdlib.h> 
#include <stdio.h> 

#define DELTA_X 1024 //This is a square of 1024
#define DELTA_Y 1024 //    by 1024

#define MAX_STEPS_KERNEL 10 //number of computations in kernel per cycle
#define MIN_VARIATION 0.05

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
			if (row == (rows/2) && col == (cols/2)){
				current_mat[(row)*cols + (col)] = 100;
			}
			else if (row == 0 && col == 0) {
				current_mat[(row)*cols + (col)] = (23 + prev_mat[(row + 1)*cols + (row)] + 23 + prev_mat[(row)*cols + (col + 1)]) / 4;
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

/*__global__ void debug_step(double *mat, size_t cols) {
	int new_step = step;
	int row = threadIdx.y;
	int col = threadIdx.x;
	double tmp = mat[(row)*cols + (col)];
}*/

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
	mat1[(rows/2)*cols + (cols/2)] = K;
	return 0;
}

bool sequential_diffuse(double *current_mat, double *prev_mat, size_t rows, size_t cols, size_t *steps){
	unsigned int is_finished = 1;
	int col = 0;
        int row = 0;
        for (row = 0; row < rows; row++){
		for(col=0; col < cols; col++){
			if (row == (rows/2) && col == (cols/2)){
                                current_mat[(row)*cols + (col)] = 100;
                        }
			else if (row == 0 && col == 0) {
        	        	current_mat[(row)*cols + (col)] = (100 + prev_mat[(row + 1)*cols + (row)] + 100 + prev_mat[(row)*cols + (col + 1)]) / 4;
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
		}
	}
	
	for (row = 0; row < rows; row++){
                for(col=0; col < cols; col++){
			unsigned int stop = 0;
			if(current_mat[(row)*cols + (col)] > prev_mat[(row)*cols + (col)]-MIN_VARIATION && current_mat[(row)*cols + (col)] < prev_mat[(row)*cols + (col)]+MIN_VARIATION) {
                                stop  = 1;
                        }
			is_finished = is_finished&stop;
		}
	}
	*steps=*steps + 1;
	//printf("From the seq: stops?: %d ",is_finished);
     
	if(is_finished == 1){
		return true;
	}
        return false;

}

void write_matrix(double *mat, size_t rows, size_t cols, char* file_name) {
	FILE *f = fopen(file_name, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	int i;
	int j;
	fprintf(f, "{\"z\": [");
	for (i = 0; i < rows; i++) {
		fprintf(f, "[");
		for (j = 0; j < cols; j++) {
			if (j == cols - 1) {
				fprintf(f, " %.5f ", mat[i*cols + j]);
			}
			else {
				fprintf(f, " %.5f, ", mat[i*cols + j]);
			}
			
		}
		if (i == rows - 1) {
			fprintf(f, "]\n");
		}
		else {
			fprintf(f, "],\n");
		}
		
	}
	fprintf(f, "] }");
	fclose(f);
}

struct timeval  tp1, tp2;

int main(int argc, char *argv[]) 
{ 
	long total_steps = 0;
  	// setup/initialize
  	if (argc != 2) {
    		printf ("usage: progName <steps>\n");
    		exit(-1);
  	} else {
    		total_steps = atol(argv[1]);
	}

	double *mat1;
	double *mat2;
	
	mat1 = (double*)calloc(DELTA_X * DELTA_Y,sizeof(double));
	mat2 = (double*)calloc(DELTA_X * DELTA_Y,sizeof(double));
	
	init_t0(mat1, DELTA_X, DELTA_Y, 1000.0, 23.0);

	//sequential_diffuse
        gettimeofday(&tp1, NULL);
	printf("Starting sequential\n");
	bool stop = false;
        bool is_swap = false;
	size_t seq_steps = 0;
	while(seq_steps < total_steps){
	//while(!stop){
		if(is_swap){
			stop = sequential_diffuse(mat1, mat2, DELTA_X, DELTA_Y, &seq_steps);
		} 
		else {
			stop = sequential_diffuse(mat2, mat1, DELTA_X, DELTA_Y, &seq_steps);
		}
		//printf("Step %d \n",seq_steps);
		is_swap = !is_swap;
	}
	gettimeofday(&tp2, NULL);
	double seq_time_result = (double) (tp2.tv_usec - tp1.tv_usec) / 1000000 + (double) (tp2.tv_sec - tp1.tv_sec);

	printf("Sequential finished in %d steps, writiing to file\n",seq_steps);
	// Allocates storage
	char *file_name = (char*)malloc(13 * sizeof(char));
	// Prints "Hello world!" on hello_world
	sprintf(file_name, "./seq_data_t%d.json",total_steps);
	if(is_swap){
		write_matrix(mat1, DELTA_X, DELTA_Y, file_name);
	}else{
		write_matrix(mat2, DELTA_X, DELTA_Y, file_name);
	}

	double *mat1_d, *mat2_d;
	int *result_d;
	printf("Allocating CUDA memory \n");

	init_t0(mat1, DELTA_X, DELTA_Y, 1000.0, 23.0);
	gettimeofday(&tp1, NULL);

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
	printf("Starting Kernel \n");
	unsigned int time_steps;
	for (time_steps = 0; time_steps < seq_steps; time_steps++) {
		if(time_steps%2 == 0){
			calc_diffuse << < dimgrid, dimblock >> > (mat2_d, mat1_d, DELTA_X, DELTA_Y, result_d);
		}
		else {
			calc_diffuse << < dimgrid, dimblock >> > (mat1_d, mat2_d, DELTA_X, DELTA_Y, result_d);
		}
		
		//debug_step <<<1, 1 >>> (mat2_d, DELTA_X);
	}

	printf("Kernel Finished \n");
	cudasafe(cudaMemcpy(mat1, mat1_d, DELTA_X * DELTA_Y * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy host <-dev(mat1_d) failed.");
	cudasafe(cudaMemcpy(mat2, mat2_d, DELTA_X * DELTA_Y * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy host <-dev(mat2_d) failed.");

	gettimeofday(&tp2, NULL);

	double par_time_result = (double) (tp2.tv_usec - tp1.tv_usec) / 1000000 + (double) (tp2.tv_sec - tp1.tv_sec);
        
	cudaDeviceSynchronize();
	printf("Writing to file \n");
	sprintf(file_name, "./par_data_t%d.json",total_steps);
	if(is_swap){
                write_matrix(mat1, DELTA_X, DELTA_Y, file_name);
        }else{
                write_matrix(mat2, DELTA_X, DELTA_Y, file_name);
        }
	printf("Finished writing \n");
	
	printf("Problem size: %d \n",DELTA_X*DELTA_Y);
	printf("seq_time, %.5f\n", seq_time_result);
	printf("par_time, %.5f\n", par_time_result);
	printf("speed_up, %.5f\n", seq_time_result/par_time_result);
	//printf("Press any key to exit!");
	//getchar();
 
	return 0; 
}
