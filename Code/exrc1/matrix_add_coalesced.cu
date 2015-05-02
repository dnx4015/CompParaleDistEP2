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
#define INC(i,n,inc) for(int i=0;i<n;i+=inc)
#define imin(a,b) (a<b?a:b)

const int BlockSizeX = 32;
const int Factor = 4;
const int BlockSizeY = BlockSizeX/Factor;

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG) 
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
			cudaGetErrorString(result));
  }
#endif
  return result;
}

__global__ void add( int* matrix_result, 
                      int* matrix_a, 
                      int* matrix_b, 
                      const int N ) {
	int col = blockIdx.x * BlockSizeX + threadIdx.x;
    int row = blockIdx.y * BlockSizeX + threadIdx.y;
	
	INC(i, BlockSizeX, BlockSizeY){
		matrix_result[(row + i) * N + col] = 
			matrix_a[(row + i) * N + col] +
			matrix_b[(row + i) * N + col];
	}
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
      fprintf(stderr, "Syntax: %s <vector size N>\n", argv[0]);
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


	//const int N = 32;
	const int mSize = N*N*sizeof(int);
	int gridSize = N / BlockSizeX;

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

	dim3 dimGrid(gridSize, gridSize, 1);
	dim3 dimBlock(BlockSizeX, BlockSizeX/Factor, 1);

    add<<<dimGrid,dimBlock>>>(dev_result, dev_a, dev_b, N);
	
	cudaDeviceSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error!=cudaSuccess) {
	   	fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
		exit(-1);
	}
    // copy the array 'result' back from the GPU to the CPU
    checkCuda( cudaMemcpy( host_result, dev_result, mSize, D2H) );

	REP(i, N){
		REP(j, N){
			DEBUG printf("(%d) ", i*(mul+1));
			DEBUG printf("%d ", host_result[i][j]);
			assert(host_result[i][j] == i*(mul+1));	
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
	printf("Done\n");
}
