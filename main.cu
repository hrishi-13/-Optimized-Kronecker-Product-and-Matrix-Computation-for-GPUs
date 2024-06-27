/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-1
 * Description: Computation of a matrix C = Kronecker_prod(A, B.T)
 *              where A and B are matrices of dimension (m, n) and
 *              the output is of the dimension (m * n, m * n). 
 * Note: All lines marked in --> should be replaced with code. 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    long int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < m * m) {
        // printf("%ld ", id);
        long int i = id / m;
        long int j = id % m;
        // printf("%ld, %ld\n", i, j);

        for(long int p = 0; p < n; p++) {
            for(long int q = 0; q < n; q++) {
                C[(n*i + q)* m*n + m*p + j] = A[i*n + p] * B[j*n + q];
                // C[n*i + q][m*p + j] = A[i][p] * B[j][q] 
            }
        }
    }
}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    long int id = (blockIdx.x * 32 * 32) + (threadIdx.y * 32) + threadIdx.x;
    if(id < n*n) { 
        // printf("%ld ", id);
        long int i = id / n;   
        long int j = id % n;
        // printf("%ld, %ld\n", i, j);

        for(long int p = 0; p < m; p++) {
          for(long int q = 0; q < m; q++) {
            C[(n*p + j)*m*n + m*i + q] = A[p*n + i] * B[q*n + j];
            // C[n*p + j][m*i + q] = A[p][i] * B[q][j]
          }
        }
    }
}

__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    long int id = (gridDim.y * blockDim.x * blockDim.y * blockIdx.x) + (blockDim.x * blockDim.y * blockIdx.y) + (threadIdx.x * blockDim.y) + threadIdx.y;
    if(id < (m*m*n*n)) {
        // printf("%ld ", id);
        long int i = id / (m*n);
        long int j = id % (m*n);
        // printf("%ld %ld\n", i, j);

        C[i*m*n + j] = A[(i/n)*n + j/m] * B[(j%m)*n + i%n];
        // C[i][j] = A[i/n][j/m] * B[j%m][i%n]
    }
}

// Prints any 1D array in the form of a matrix

void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/

double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc, char **argv){
    // Variable declarations
    long int m,n;	
    cin >> m >> n;	

    // Host_arrays 
    long int *h_a, *h_b, *h_c;

    // Device arrays 
    long int *d_a, *d_b, *d_c;
	
    /* Allocating space for the host_arrays */
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));	
    h_c = (long int *) malloc(m * m * n * n * sizeof(long int));	

    /* Allocating memory for the device arrays */
    // Allocating memory for A on device 
    cudaMalloc(&d_a, m*n*sizeof(long int));

    // Allocating memory for B on device 
    cudaMalloc(&d_b, m*n*sizeof(long int));

    // Allocating memory for C on device 
    cudaMalloc(&d_c, m*n*n*m*sizeof(long int));

    // Reading the input matrix A 
    for(long int i = 0; i < m * n; i++) {
        cin >> h_a[i];
    }

    // Reading the input matrix B 
    for(long int i = 0; i < m * n; i++) {
        cin >> h_b[i];
    }

    // Transfer the input host arrays to the device 
    // Copying A from Host to Device
    // Copying B from Host to Device 
    cudaMemcpy(d_a, h_a, m*n*sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, m*n*sizeof(long int), cudaMemcpyHostToDevice);

    long int gridDimx, gridDimy;
    
    /**
     * Kernel 1 - per_row_AB_kernel
     * To be launched with 1D grid, 1D block
     * Each thread should process a complete row of A, B
     **/

    // Setting the launch configuration 
    gridDimx =  ceil((float)(m*m)/1024);
    dim3 grid1(gridDimx, 1, 1); // 1D Grid
    dim3 block1(1024, 1, 1); // 1D Block

    double starttime = rtclock();  

    // Launching the per_row kernel 
    per_row_AB_kernel<<<grid1, block1>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize();                                                           

    double endtime = rtclock(); 
	printtime("GPU Kernel-1 time: ", starttime, endtime);  

    // Copying C from Device to Host 
    cudaMemcpy(h_c, d_c, m*n*m*n*sizeof(long int), cudaMemcpyDeviceToHost);

    printMatrix(h_c, m * n, m * n,"kernel1.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));

    /**
     * Kernel 2 - per_column_AB_kernel
     * To be launched with 1D grid, 2D block
     * Each thread should process a complete column of  A, B
     **/
    
    // Setting the launch configuration 
    gridDimx =  ceil((float)(n*n)/1024);
    dim3 grid2(gridDimx, 1, 1); // 1D Grid
    dim3 block2(32, 32, 1); // 2D Block

    starttime = rtclock(); 

    // Launching the per_column kernel 
    per_column_AB_kernel<<<grid2, block2>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize(); 

    endtime = rtclock(); 
  	printtime("GPU Kernel-2 time: ", starttime, endtime);  

    // Copying C from Device to Host
    cudaMemcpy(h_c, d_c, m*n*m*n*sizeof(long int), cudaMemcpyDeviceToHost);

    printMatrix(h_c, m * n, m * n,"kernel2.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(long int));

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     * Each thread should process one element of the output 
     **/

    // Setting the launch configuration 
    gridDimx = ceil(float(n * n) / 16);
    gridDimy = ceil(float(m * m) / 64);
    dim3 grid3(gridDimx,gridDimy,1); // 2D Grid
    dim3 block3(64,16,1); // 2D Block

    starttime = rtclock();  

    // Launching the per_element kernel 
    per_element_kernel<<<grid3, block3>>>(d_a, d_b, d_c, m, n);
    cudaDeviceSynchronize();                                                              

    endtime = rtclock();  
	printtime("GPU Kernel-3 time: ", starttime, endtime);  

    // Copying C from Device to Host
    cudaMemcpy(h_c, d_c, m*n*m*n*sizeof(long int), cudaMemcpyDeviceToHost);
    printMatrix(h_c, m * n, m * n, "kernel3.txt");

    return 0;
}