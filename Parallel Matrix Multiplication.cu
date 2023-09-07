#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;
__global__ void firstMultiplication(int *A,int *B,int *C,int p,int q,int r){
	__shared__ int rowManipulation[1024];
	int i = blockIdx.x;
	int j = threadIdx.x;
	if(j==(r-1))
	{	
		for(int s=0;s<1024;s++)
		{
			rowManipulation[s]=0;
	    }
  	}
  	__syncthreads();

	for(int k=0;k<q;k++){
		rowManipulation[j] = rowManipulation[j] + A[i*q+k] * B[k*r+j];
		__syncthreads();
	}
	__syncthreads();
	if(j==(r-1))
		for(int cpy=0;cpy<r;cpy++){
			C[blockIdx.x*r+cpy] = rowManipulation[cpy];	
		}
}


__global__ void secondMultiplication(int *C,int *DT,int *result,int p,int q,int r){
	int k = threadIdx.x;
	int finalRow = blockIdx.x;
	int Column = blockIdx.y;

	__shared__ int temp[1024];

	if(k==(q-1)){
		for(int a=0;a<1024;a++){
			temp[a]=0;
		}
	}
	__syncthreads(); 
	temp[k] = C[finalRow*q+k]*DT[Column*q+k];
	__syncthreads();

	if(k==(q-1)){
		result[finalRow*r+Column]=0;
		for(int a=0;a<q;a++){
			result[finalRow*r+Column]+=temp[a];
		}
	}
}

__global__ void finalAddition(int *AB,int *CDT,int *result,int p,int q,int r){
	int i = blockIdx.x;
	int j = threadIdx.x;
	int id = i*r+j;
	result[id] = AB[id] + CDT[id];
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	int *AB,*CDT;
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));
	cudaMalloc(&AB,p*r*sizeof(int));
	cudaMalloc(&CDT,p*r*sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */

	//A and B multiplication kernel call and their multiplication store on the AB GPU Memory

	firstMultiplication<<<p,r>>>(d_matrixA,d_matrixB,AB,p,q,r);

	//C and D transpose kernel call and their multiplication store on the CDT GPU memory
	dim3 grid(p,r,1);
	secondMultiplication<<<grid,q>>>(d_matrixC,d_matrixD,CDT,p,q,r);	

	// Addition of AB and CDT using a kernel call and store it inside the d_matrixE in GPU memory

	finalAddition<<<p,r>>>(AB,CDT,d_matrixE,p,q,r);


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
	
