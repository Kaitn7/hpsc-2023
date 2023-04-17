#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0);

  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  
  for(int i=0; i<range; i++)
    printf("%d ", bucket[i]);
  printf("\n");

  std::vector<int> offset(range,0), temp(range,0);

#pragma omp parallel
  for (int j=1; j<range; j<<=1) {
#pragma omp for
    for(int i=0; i<range; i++)
      temp[i] = offset[i];
#pragma omp for
    for (int i=j; i<range; i++){
      offset[i] += temp[i-j] + bucket[i-j];
    }
  }

  // for (int i=0; i<range; i++)
  //   // offset[i] = offset[i-1] + bucket[i-1];
  
#pragma omp for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
