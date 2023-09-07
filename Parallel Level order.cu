/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

//Kernel calling for level calculation of node
/*   Number of threads call in this kernel ius equal to number of nodes in a particular level 
       of graph for which we are calucating level value of node.
       As we know in  a CSR list is in sorted order so we find a maximum vertex that is connected to 
         nodes of that level.

*/

__global__ void LevelCalculation(int *d_offset,int *d_csrList,int *dmax,int calls,int m)
{
  //Global ID calculation for every thread
     int globalId = blockIdx.x*blockDim.x + threadIdx.x;

  //condition check for number of threads less than number of nodes in that level
    if(globalId < calls){
        int node = globalId + m;
        if(d_offset[node]!=d_offset[node+1])
        {
           int lmax = d_csrList[d_offset[node+1]-1];
           //Atomic Comparision of maximum connected node calculation. Atomic will handle data race 
           atomicMax(dmax,lmax);
        }
        
    }
}
          
//Kernel calling for active indegree calculation of every vertex of that level.
/*   from a active node vertex outgoing edges will be the active indegree of that connected vertex.
         Here we will use atomic increment for handle the data race between different threads.

*/
__global__ void IndegreeCalculation(int *d_offset,int *d_csrList,int calls,int start,int end,unsigned int *d_indegree,unsigned int *d_activation,int V)
{
  //Global Id calculation for respective thread
  int globalId = blockIdx.x*blockDim.x + threadIdx.x;
  //condition check for number of threads less than number of nodes in that level
  if(globalId < calls){
      int node = globalId + start;
    if(((int)d_activation[node])==1 && node<(V-1) && d_offset[node]!=d_offset[node+1]){
      int no_of_edges = d_offset[node+1] - d_offset[node];
      int p = d_offset[node];
      for(int i=1;i<=no_of_edges;i++){
        int ed = d_csrList[p++];
        atomicInc(&d_indegree[ed],10000);
      }
    }  
  }
}    

//Kernel calling for finding active node by comparing the APR value and active indegree 
__global__ void ActivationComparision(unsigned int *d_indegree,unsigned int *d_activation,int *d_apr,int calls,int start,int end)
{
  //Global Id calculation for respective thread
  int globalId = blockIdx.x*blockDim.x + threadIdx.x;
  //condition check for number of threads less than number of nodes in that level
  if(globalId < calls){
    int node = globalId + start;
    if(((int)d_indegree[node])>=d_apr[node]){
      d_activation[node]=1;
    }
  }
}


//Kernel calling for finding Deactivated node by the adjacent vertices
__global__ void DeactivationCheck(unsigned int *d_activation,int no_of_calls,int start,int end)
{
  //Global Id calculation for respective thread
  int globalId = blockIdx.x*blockDim.x + threadIdx.x;
  //condition check for number of threads less than number of nodes in that level
  if(globalId < no_of_calls){
    int node = globalId + start;
    //condition checking for activated node
    if(node!=start && node!=end){
      if(d_activation[node-1]==0 && d_activation[node+1]==0)
        d_activation[node]=0;
    }
  }
}

//Kernel calling for final result calculation by calculating number of the active nodes on the level

__global__ void ResultCalculation(unsigned int *d_activation,unsigned int *d_result,int no_of_calls,int start)
{
  //Global Id calculation for respective thread
  int globalId = blockIdx.x*blockDim.x + threadIdx.x;
  //condition check for number of threads less than number of nodes in that level
  if(globalId < no_of_calls){
    int node = globalId + start;
    //condition checking that node is activated or not
    if(d_activation[node]==1){
        atomicInc(&d_result[0],10000);
    }
  }
}
    
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
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

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    // int *d_activeVertex;
	// cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/


//Level calculation 
/* for every level calculate the starting node and ending node of the level  
*/


int *levelstart,*levelend;
int *hmax,*dmax;
hmax = (int*)malloc(sizeof(int));
levelstart = (int*)malloc(L*sizeof(int));
levelend = (int*)malloc(L*sizeof(int));
cudaMalloc(&dmax,1*sizeof(int));
levelstart[0]=0;
int i = 0;
while(h_apr[i]==0){
  i = i+1;
}
i = i-1;
cudaMemset(dmax,0, sizeof(int));
levelend[0] = i;

//kernel calling for the level calculation level by level
for(int j=1;j<L;j++){
    int calls = levelend[j-1]-levelstart[j-1]+1;
    int blocks = ceil((float)calls/1024);
    levelstart[j]=levelend[j-1]+1;
    LevelCalculation<<<blocks,1024>>>(d_offset,d_csrList,dmax,calls,levelstart[j-1]);
    cudaMemcpy(hmax, dmax, sizeof(int), cudaMemcpyDeviceToHost);
    levelend[j] = *hmax;
}




unsigned int *d_indegree,*h_activation,*d_activation;
cudaMalloc(&d_indegree,V*sizeof(unsigned int));
cudaMalloc(&d_activation,V*sizeof(unsigned int));
h_activation = (unsigned int*)malloc(V*sizeof(unsigned int));


cudaMemset(d_indegree,0, V*sizeof(unsigned int));
cudaMemset(d_activation,0, V*sizeof(unsigned int));
memset(h_activation, 0, V*sizeof(unsigned int));

//Initialization of level zero arrays because activation of the level zero array is always 1.
for(int i = 0;i<=levelend[0];i++){
  h_activation[i]=1;
}
cudaMemcpy(d_activation,h_activation, (levelend[0]+1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
h_activeVertex[0] = levelend[0]+1;

/*for loop for calculating indegree , activation of the node and deactivation function call 
  level by level.
*/

for(int k=0;k<(L-1);k++){


  //Indegree calculation kernel call
  int no_of_calls = levelend[k] - levelstart[k] + 1;
  IndegreeCalculation<<<ceil((float)no_of_calls/1024),1024>>>(d_offset,d_csrList,no_of_calls,levelstart[k],levelend[k],d_indegree,d_activation,V);
  cudaDeviceSynchronize();

  //Activation kernel comparision for k+1 level
  no_of_calls = levelend[k+1]-levelstart[k+1]+1;
  ActivationComparision<<<ceil((float)no_of_calls/1024),1024>>>(d_indegree,d_activation,d_apr,no_of_calls,levelstart[k+1],levelend[k+1]);
  cudaDeviceSynchronize();


  //Deactivation kernel call for k+1 level kernel call
  
  no_of_calls = levelend[k+1]-levelstart[k+1]+1;
  DeactivationCheck<<<ceil((float)no_of_calls/1024),1024>>>(d_activation,no_of_calls,levelstart[k+1],levelend[k+1]);
  cudaDeviceSynchronize();


  

}

unsigned int *h_result,*d_result;
h_result = (unsigned int*)malloc(sizeof(unsigned int));
cudaMalloc(&d_result,sizeof(unsigned int));

//final result calculation and storing the result into h_activeVertex

  for(int k=1;k<L;k++){
    cudaMemset(d_result,0,sizeof(unsigned int));
    memset(h_result, 0, sizeof(unsigned int));
    int no_of_calls = levelend[k] - levelstart[k] + 1;
    ResultCalculation<<<ceil((float)no_of_calls/1024),1024>>>(d_activation,d_result,no_of_calls,levelstart[k]);
    cudaMemcpy(h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    h_activeVertex[k] = (int)*h_result;
  }
   
    
    

     

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
