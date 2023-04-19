#include <cstdio>
#include <cstdlib>
#include <vector>

void para_prf_sum(int *offset, int size){
  int *temp;
  temp = new int[size];
#pragma omp parallel
  {
  for(int j=1; j<size; j<<=1) {
#pragma omp for
    for(int i=0; i<size; i++)
      temp[i] = offset[i];
#pragma omp for
    for(int i=j; i<size; i++)
      offset[i] += temp[i-j];
    }
  }
  delete[] temp;
}

int main() {
  int n = 20;
  int range = 10;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  bucket = new int[range];

#pragma omp parallel for schedule(auto) reduction(+:bucket[:range])
    for (int i=0; i<n; i++){
      bucket[key[i]]++;
    }
  
  for(int i=0; i<range; i++)
    printf("%d ", bucket[i]);
  printf("\n");
  int *offset;
  offset = new int[range];
  offset[0] = 0;
#pragma omp parallel for
  for (int i=0; i<range-1; i++){
    offset[i+1] = bucket[i];
  }
  para_prf_sum(offset, range);

  // for (int i=0; i<range; i++)
  //   // offset[i] = offset[i-1] + bucket[i-1];

  for(int i=0; i<range; i++)
    printf("%d ", offset[i]);
  printf("\n");
  
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
