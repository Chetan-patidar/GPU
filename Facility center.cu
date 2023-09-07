#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define max_N 100000
#define max_P 30
#define BLOCKSIZE 1024

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void calculateSuccess(int *d_req_id,int *d_req_cen,int *d_req_fac,int *d_req_start,int *d_req_slots,int *d_initial_centre,int *d_capacity,int *d_initial_req,int *d_succ_req,int *d_facility, int N){

  int index =  threadIdx.x + blockDim.x * blockIdx.x;
  if(index < N){
    int arr[30*24];
    
    for(int i = 0; i < (d_facility[index]*24); i++)
      arr[i] = 0;

    for(int i = d_initial_req[index]; i < d_initial_req[index+1]; i++){
      int start = d_req_fac[i]*24 + d_req_start[i]-1;
      int z = 0;
      int end = start + d_req_slots[i];
      int j;
      
    
      for(j = start; j < end; j++){
        int x = d_capacity[d_initial_centre[index]+d_req_fac[i]];
        int y = d_req_fac[i]*24 + 24;
        if(arr[j] < x && end <= y)
          continue;
        else{
          break;
        }
      }
      
      if(j == end){
        for(j = start; j < end; j++){
          arr[j]++;
        }
        d_succ_req[index]++;
      }
    }
  }
}


//***********************************************
//Merge sort function declaration for sorting the array when we required

//merge function for merging the two sorted array
void merge(int start,int mid,int end,int *req_id,int *req_cen,int *req_fac,int *req_start,int *req_slots,int R){
  
  //intermediate array declaration 
  int *arr1,*arr2,*arr3,*arr4,*arr5;
  arr1 = (int*)malloc((R)*sizeof(int));
  arr2 = (int*)malloc((R)*sizeof(int));
  arr3 = (int*)malloc((R)*sizeof(int));
  arr4 = (int*)malloc((R)*sizeof(int));
  arr5 = (int*)malloc((R)*sizeof(int));

//merging the arrays
  int current=0,left = start,right = mid+1;
  while (left <= mid && right <= end) {
        if (req_cen[left] < req_cen[right]) {
            arr1[current] = req_id[left];
            arr2[current] = req_cen[left];
            arr3[current] = req_fac[left];
            arr4[current] = req_start[left];
            arr5[current] = req_slots[left];
            left++;
        }
        else if(req_cen[left]==req_cen[right]){
            if(req_id[left]<req_id[right]){
                arr1[current] = req_id[left];
                arr2[current] = req_cen[left];
                arr3[current] = req_fac[left];
                arr4[current] = req_start[left];
                arr5[current] = req_slots[left];
                left++;
            }
            else{
                arr1[current] = req_id[right];
                arr2[current] = req_cen[right];
                arr3[current] = req_fac[right];
                arr4[current] = req_start[right];
                arr5[current] = req_slots[right];
                right++;
            }
        }
        else{
                arr1[current] = req_id[right];
                arr2[current] = req_cen[right];
                arr3[current] = req_fac[right];
                arr4[current] = req_start[right];
                arr5[current] = req_slots[right];
                right++;
        }
        current++;
    }


    while (left <= mid) {
        arr1[current] = req_id[left];
        arr2[current] = req_cen[left];
        arr3[current] = req_fac[left];
        arr4[current] = req_start[left];
        arr5[current] = req_slots[left];
        left++;
        current++;
    }

    while (right <= end) {
        arr1[current] = req_id[right];
        arr2[current] = req_cen[right];
        arr3[current] = req_fac[right];
        arr4[current] = req_start[right];
        arr5[current] = req_slots[right];
        right++;
        current++;
    }


    int p = 0;
    for(int i=start;i<=end;i++){
      req_id[i] = arr1[p];
      req_cen[i] = arr2[p];
      req_fac[i] = arr3[p];
      req_start[i] = arr4[p];
      req_slots[i] = arr5[p];
      p++;
    }

}


//merge sort main function
void rearrange(int start,int end,int *req_id,int *req_cen,int *req_fac,int *req_start,int *req_slots,int R){

  if (start >= end){
    return;
  }
  int mid = (start + end)/2;
  rearrange(start,mid,req_id,req_cen,req_fac,req_start,req_slots,R);
  rearrange(mid+1,end,req_id,req_cen,req_fac,req_start,req_slots,R);
  merge(start,mid,end,req_id,req_cen,req_fac,req_start,req_slots,R);
  
}









int main(int argc,char **argv)
{
	// variable declarations...
    int N,*centre,*facility,*capacity,*fac_ids, *succ_reqs, *tot_reqs;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &N ); // N is number of centres
	
    // Allocate memory on cpu
    centre=(int*)malloc(N * sizeof (int));  // Computer  centre numbers
    facility=(int*)malloc(N * sizeof (int));  // Number of facilities in each computer centre
    fac_ids=(int*)malloc(max_P * N  * sizeof (int));  // Facility room numbers of each computer centre
    capacity=(int*)malloc(max_P * N * sizeof (int));  // stores capacities of each facility for every computer centre 


    int success=0;  // total successful requests
    int fail = 0;   // total failed requests
    tot_reqs = (int *)malloc(N*sizeof(int));   // total requests for each centre
    succ_reqs = (int *)malloc(N*sizeof(int)); // total successful requests for each centre

    // Input the computer centres data
    int k1=0 , k2 = 0;
    for(int i=0;i<N;i++)
    {
      fscanf( inputfilepointer, "%d", &centre[i] );
      fscanf( inputfilepointer, "%d", &facility[i] );
      
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &fac_ids[k1] );
        k1++;
      }
      for(int j=0;j<facility[i];j++)
      {
        fscanf( inputfilepointer, "%d", &capacity[k2]);
        k2++;     
      }
    }

    // variable declarations
    int *req_id, *req_cen, *req_fac, *req_start, *req_slots;   // Number of slots requested for every request
    //variable declarations for GPU
    int *d_req_id, *d_req_cen, *d_req_fac, *d_req_start, *d_req_slots, *d_initial_centre, *d_capacity, *d_initial_req, *d_succ_req, *d_facility;
    // Allocate memory on CPU 
	int R;
	fscanf( inputfilepointer, "%d", &R); // Total requests
    req_id = (int *) malloc ( (R) * sizeof (int) );  // Request ids
    req_cen = (int *) malloc ( (R) * sizeof (int) );  // Requested computer centre
    req_fac = (int *) malloc ( (R) * sizeof (int) );  // Requested facility
    req_start = (int *) malloc ( (R) * sizeof (int) );  // Start slot of every request
    req_slots = (int *) malloc ( (R) * sizeof (int) );   // Number of slots requested for every request
    
  //Allocate memory on GPU

  cudaMalloc(&d_req_id, R * sizeof(int));
  cudaMalloc(&d_initial_req, (N+1) * sizeof(int));
  cudaMalloc(&d_req_fac, R * sizeof(int));
  cudaMalloc(&d_req_start, R * sizeof(int));
  cudaMalloc(&d_initial_centre, N * sizeof(int));
  cudaMalloc(&d_capacity, 30*N * sizeof(int));
  cudaMalloc(&d_req_slots, R * sizeof(int));
  cudaMalloc(&d_succ_req, N*sizeof(int));
  cudaMalloc(&d_facility, N*sizeof(int));
  cudaMalloc(&d_req_cen, R * sizeof(int));
  cudaMemset(d_succ_req, 0, N*sizeof(int));




    // Input the user request data
    for(int j = 0; j < R; j++)
    {
       fscanf( inputfilepointer, "%d", &req_id[j]);
       fscanf( inputfilepointer, "%d", &req_cen[j]);
       fscanf( inputfilepointer, "%d", &req_fac[j]);
       fscanf( inputfilepointer, "%d", &req_start[j]);
       fscanf( inputfilepointer, "%d", &req_slots[j]);
       tot_reqs[req_cen[j]]+=1;  
    }
		


  //Sort the request based on the centre so that we can call kernel on the basis on centre and  will run on the parallel
  int add=0,*initial_centre,*initial_req;
  initial_centre = (int*)malloc(N*sizeof(int));
  initial_req = (int*)malloc((N+1)*sizeof(int));
  int i = 0;
  initial_req[0]=0;
  while(i<N){
    initial_centre[i] = add;
    add = add + facility[i];
    initial_req[i+1] = initial_req[i]+tot_reqs[i];
    i++;
  }
 
 //sorting the request arrays on the basis of centre
  rearrange(0,R-1,req_id,req_cen,req_fac,req_start,req_slots,R);
  
//copying the cpu arrays into gpu arrays
  cudaMemcpy(d_initial_req, initial_req, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_id, req_id, R*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_fac, req_fac, R*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_start, req_start, R*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_slots, req_slots, R*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_initial_centre, initial_centre, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_facility, facility, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_req_cen, req_cen, R*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_capacity, capacity, 30*N*sizeof(int), cudaMemcpyHostToDevice);





    //*********************************
    // Call the kernels here
  //calling the calculateSuccess kernel where all the centres running in parallel
  calculateSuccess<<<ceil((float)N/1024), 1024>>>(d_req_id, d_req_cen, d_req_fac, d_req_start, d_req_slots, d_initial_centre, d_capacity, d_initial_req, d_succ_req, d_facility, N);
  cudaDeviceSynchronize();

    //********************************

 //copy the answer in a cpu array
  cudaMemcpy(succ_reqs, d_succ_req, N * sizeof(int), cudaMemcpyDeviceToHost);
    int m = 0;
    while(m<N){
      success = success + succ_reqs[m];
      m++;
    }

    fail = R - success;

    





    // Output
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    fprintf( outputfilepointer, "%d %d\n", success, fail);
    for(int j = 0; j < N; j++)
    {
        fprintf( outputfilepointer, "%d %d\n", succ_reqs[j], tot_reqs[j]-succ_reqs[j]);
    }
    fclose( inputfilepointer );
    fclose( outputfilepointer );
    cudaDeviceSynchronize();
	return 0;
}