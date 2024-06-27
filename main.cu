#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

__global__ void matrixAddition(int *d_matrixE, int *d_matrixA, int *d_matrixB, int p, int r) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = (i * r) + j;
    if ((i < p) and (j < r)) {
        d_matrixE[index] = d_matrixA[index] + d_matrixB[index];
	}
}

// kernel for matrix multiplication, C = A * B

__global__ void matrixProduct(int *d_matrixC, int *d_matrixA, int *d_matrixB, int p, int q, int r) {

    int i = (blockIdx.x * 32) + threadIdx.x;
    int j = (blockIdx.y * 32) + threadIdx.y;

	int sum = 0;

	__shared__ int shared_A[32][32+1];
    __shared__ int shared_B[32][32+1];
	// Doing 32+1 to avoid bank conflict

	shared_A[threadIdx.y][threadIdx.x] = 0;
    shared_B[threadIdx.y][threadIdx.x] = 0;

	int k = ceil((float)(q) / 32);

    for(int x = 0; x < k; x++) {

		int t_x = (x * 32) + threadIdx.x;
		int t_y = (x * 32) + threadIdx.y;

        if((j < p) and (t_x < q)){
            shared_A[threadIdx.y][threadIdx.x] = d_matrixA[t_x + (j * q)];
		}

        else{
            shared_A[threadIdx.y][threadIdx.x] = 0;
		}

        if((t_y < q) and (i < r)){
            shared_B[threadIdx.y][threadIdx.x] = d_matrixB[(t_y) * r + i];
		}

        else{
            shared_B[threadIdx.y][threadIdx.x] = 0;
		}

        __syncthreads(); 
		// Synchronizes all threads in a block 

        for(int y = 0; y < 32; y++){
            sum = sum + shared_A[threadIdx.y][y] * shared_B[y][threadIdx.x];
		}
        __syncthreads();
		// Avoids memory hazards 
    }

    if((i<r) and (j<p)){
        d_matrixC[(j*r) + i] = sum;
	}
}

// kernel to make transpose of a matrix

__global__ void matrixTranspose(int *d_matrixD_Transpose, int *d_matrixD, int r, int q) {

    int i = (blockIdx.x * 32) + threadIdx.x;
    int j = (blockIdx.y * 32) + threadIdx.y;

	__shared__ int shared_D[32][32+1];
	// Doing 32+1 to avoid bank conflict
    
    if((i<q) and (j<r)){
        shared_D[threadIdx.y][threadIdx.x] = d_matrixD[(j*q) + i];
	}

    __syncthreads();

    i = (blockIdx.y * 32) + threadIdx.x;
    j = (blockIdx.x * 32) + threadIdx.y;

    if((i<r) and (j<q)){
        d_matrixD_Transpose[(j*r) + i] = shared_D[threadIdx.x][threadIdx.y];
	}
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...

	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	
	int gridDimx, gridDimy;
	// Device variables declarations
	int *d_matrixTemp, *d_matrixTemp1, *d_matrixTemp2; 
	// allocating memory
	cudaMalloc(&d_matrixTemp, q * r * sizeof(int));
	cudaMalloc(&d_matrixTemp1, p * r * sizeof(int));
	cudaMalloc(&d_matrixTemp2, p * r * sizeof(int));

	/* Matrix Transpose */
	gridDimx = ceil((float)(q) / 32);
	gridDimy = ceil((float)(r) / 32);
	dim3 grid1(gridDimx, gridDimy, 1); 
    dim3 block1(32, 32, 1); 
	// Temp = D^T
    matrixTranspose<<<grid1, block1>>>(d_matrixTemp, d_matrixD, r, q);
    cudaFree(d_matrixD);
	// D = Temp
	d_matrixD = d_matrixTemp;

	/* Matrix Multiplication */
	gridDimx = ceil((float)(r) / 32);
	gridDimy = ceil((float)(p) / 32);
	dim3 grid2(gridDimx, gridDimy, 1); 
    dim3 block2(32, 32, 1);

	// Temp1 = A * B
    matrixProduct<<<grid2, block2>>>(d_matrixTemp1, d_matrixA, d_matrixB, p, q, r);
    
	// Temp2 = C * D
    matrixProduct<<<grid2, block2>>>(d_matrixTemp2, d_matrixC, d_matrixD, p, q, r);
	
	/* Matrix Addition */
	gridDimx = ceil((float)(p) / 32);
	gridDimy = ceil((float)(r) / 32);
    dim3 grid3(gridDimx, gridDimy);
	dim3 block3(32, 32);

	// E = Temp1 + Temp2
    matrixAddition<<<grid3, block3>>>(d_matrixE, d_matrixTemp1, d_matrixTemp2, p, r);

	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));

	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}