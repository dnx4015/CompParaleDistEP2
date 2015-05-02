#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define uint unsigned int
#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define DEBUG if(0)

#define REP(i,n) for(int i=0;i<n;i++)
#define FOR(i,a,b) for(int i=a;i<=b;i++)
#define imin(a,b) (a<b?a:b)

#define BLOCK_SIZE 16

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG) 
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
			cudaGetErrorString(result));
  }
#endif
  return result;
}

__global__ void mult( int* matrix_result, 
                      int* matrix_a, 
                      int* matrix_b, 
                      int N ) {
    int val = 0;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	REP(i, N){
		val += matrix_a[col * N + i] * matrix_b[i * N + row];
	}
	matrix_result[col * N + row] = val;
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
      fprintf(stderr, "Syntax: %s <vector size N (<=512)>\n", argv[0]);
      return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);

	if (N > 512) {
      fprintf(stderr, "Syntax: %s <vector size N (<=512)>\n", argv[0]);
      return EXIT_FAILURE;
    }

	cudaEvent_t beginEvent ;
	cudaEvent_t endEvent ;

	cudaEventCreate( &beginEvent ) ;
	cudaEventCreate( &endEvent ) ;

	cudaEventRecord( beginEvent , 0 ) ;

	//const int N = 4;
	int mSize = N * N * sizeof(int);
	int col_sum = N * (N - 1) / 2;
	int mul = 5;

	int host_a[N][N], host_b[N][N], host_result[N][N];
	int *dev_a, *dev_b, *dev_result;

 	REP(i, N){
		REP(j, N){
			host_a[i][j] = i * mul;
			host_b[i][j] = i;
		}
	}

    // allocate the memory on the GPU
    checkCuda( cudaMalloc( (void**)&dev_a, mSize ) );
    checkCuda( cudaMalloc( (void**)&dev_b, mSize ) );
    checkCuda( cudaMalloc( (void**)&dev_result, mSize));

    // copy the arrays 'a' and 'b' to the GPU
    checkCuda(cudaMemcpy(dev_a, host_a, mSize, H2D));
    checkCuda(cudaMemcpy(dev_b, host_b, mSize, H2D));

	int gridSize = N / BLOCK_SIZE;
	dim3 dimGrid(gridSize, gridSize, 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    mult<<<dimGrid,dimBlock>>>(dev_result, dev_a, dev_b, N );
	
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess) {
	   	fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
		exit(-1);
	}
    // copy the array 'result' back from the GPU to the CPU
    checkCuda( cudaMemcpy( host_result, dev_result, mSize, D2H )); 

	REP(i, N){
		REP(j, N){
			DEBUG printf("(%d) ", i*mul*col_sum);
			DEBUG printf("%d ", host_result[i][j]);
			assert(host_result[i][j] == i*mul*col_sum);	
		}	
		DEBUG printf("\n");
	}

    // free memory on the gpu side
    checkCuda( cudaFree( dev_a ) );
    checkCuda( cudaFree( dev_b ) );
    checkCuda( cudaFree( dev_result ) );

	cudaEventRecord( endEvent , 0 ) ;
	cudaEventSynchronize( endEvent ) ;

	float timeValue ;
	cudaEventElapsedTime( &timeValue , beginEvent , endEvent ) ;
	printf( "Time: %.2fs\n" , timeValue ) ;

}
