#include <cstdio>
#include <cstdlib>
#include <vector>

void para_prf_sum(int *arr, int *offset){
  int N = sizeof(arr);
  int *temp;
  temp = new int[N];
#pragma omp parallel
  { 
#pragma omp for
  for (int i=0; i<N; i++){
    offset[i] = 0;
    printf("%d ", offset[i]);
  }
    
  for(int j=1; j<N; j<<=1) {
#pragma omp for
  for(int i=0; i<N; i++)
    temp[i] = offset[i];
#pragma omp for
  for(int i=j; i<N; i++)
      offset[i] += temp[i-j] + arr[i-j];
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  bucket = new int[range];

#pragma omp parallel for reduction(+:bucket[:range])
    for (int i=0; i<n; i++){
      bucket[key[i]]++;
    }
  
  for(int i=0; i<range; i++)
    printf("%d ", bucket[i]);
  printf("\n");
  int *offset;
  offset = new int[range];

  para_prf_sum(bucket, offset);

  // for (int i=0; i<range; i++)
  //   // offset[i] = offset[i-1] + bucket[i-1];
  
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++){
    printf("%d ",key[i]);
  }
  printf("\n");
}
