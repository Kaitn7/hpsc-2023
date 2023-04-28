#include <cstdio>
#include <cstdlib>


__global__ void init_zero(int *a, const int size){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i<size) a[i] = 0;
}

__global__ void add_bucket(int *bucket, int *key, const int size, const int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  extern __shared__ int tmp[];
  __syncthreads();
  if (threadIdx.x<range) tmp[threadIdx.x] = 0;
  __syncthreads();

  int stride = blockDim.x * gridDim.x;
  for (int j=i; j<size; j+=stride)
    atomicAdd(&(tmp[key[j]]), 1);
  
  __syncthreads();

  if (threadIdx.x<range)
    atomicAdd(&bucket[threadIdx.x],tmp[threadIdx.x]);
}

__global__ void scan(int *offset, int *bucket, int *buffer, const int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x+1;
  if(i>=size) return;
  offset[i] = bucket[i-1];
  for(int j=1; j<size; j<<=1) {
    buffer[i] = offset[i];
    __syncthreads();
    offset[i] += buffer[i-j];
    __syncthreads();
  } 
}

__global__ void sort_key(int *bucket, int *offset, int *key, const int range){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=range) return;
  int j = offset[i];
  for (; bucket[i]>0; bucket[i]--) {
    key[j++] = i;
  }
}

int main() {
  const int n = 50;
  const int range = 5;

  const int M = 1024;

  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  
  int *bucket;
  cudaMallocManaged(&bucket, range*sizeof(int));
  init_zero<<<(n+M-1)/M+1,M>>>(bucket,range);
  cudaDeviceSynchronize();

  add_bucket<<<(n+M-1)/M,M,range>>>(bucket, key, n, range);
  cudaDeviceSynchronize();
  // for (int i=0; i<range; i++)
  //   printf("%d ", bucket[i]);
  // printf("\n");


  int *offset, *buffer;
  cudaMallocManaged(&offset, range*sizeof(int));
  cudaMallocManaged(&buffer, range*sizeof(int));

  scan<<<(range+M-1)/M+1,M>>>(offset, bucket, buffer, range);
  cudaDeviceSynchronize();
  cudaFree(buffer);

  // for (int i=0; i<range; i++)
  //   printf("%d ", offset[i]);
  // printf("\n");


  sort_key<<<(range+M-1)/M+1,M>>>(bucket,offset,key,range);
  cudaDeviceSynchronize();

  // for (int i=0; i<n; i++)
  //   printf("%d ",key[i]);
  // printf("\n");

  for (int i = 0; i < n; i++) {
    printf("%d ",key[i]);
    if (i != 0){
      if (key[i] < key[i - 1] ) {
        printf("Sorting failed.\n");
        break;
      }
    }
    
  }
  // printf("Sorting succeeded.\n");


  cudaFree(offset);
  cudaFree(bucket);
  
}
