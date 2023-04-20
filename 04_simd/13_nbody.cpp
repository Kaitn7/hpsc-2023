#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

__m256 intr_reduct(__m256 vec){
  __m256 sum = _mm256_permute2f128_ps(vec,vec,1);
  sum = _mm256_add_ps(sum, vec);
  sum = _mm256_hadd_ps(sum, sum);
  sum = _mm256_hadd_ps(sum, sum);
  return sum;
}


int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], ind[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    ind[i] = i;
  }

  __m256 indvec = _mm256_load_ps(ind);

  __m256 x_vec = _mm256_load_ps(x);
  __m256 y_vec = _mm256_load_ps(y);
  __m256 m_vec = _mm256_load_ps(m);

  __m256 zerovec = _mm256_setzero_ps();

  for(int i=0; i<N; i++) {
    __m256 xi_vec = _mm256_set1_ps(x[i]);
    __m256 yi_vec = _mm256_set1_ps(y[i]);

    __m256 const ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(indvec, ivec, _CMP_EQ_OQ);
    
    __m256 rx = _mm256_sub_ps(xi_vec, x_vec);
    __m256 ry = _mm256_sub_ps(yi_vec, y_vec);

    __m256 inv_r = _mm256_blendv_ps(_mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rx,rx),_mm256_mul_ps(ry,ry))), zerovec, mask);

    __m256 temp = _mm256_mul_ps(m_vec,_mm256_mul_ps(inv_r,_mm256_mul_ps(inv_r,inv_r)));

    float sum[N];
    _mm256_store_ps(sum, intr_reduct(_mm256_mul_ps(rx,temp)));
    fx[i] -= sum[0];

    _mm256_store_ps(sum, intr_reduct(_mm256_mul_ps(ry,temp)));
    fy[i] -= sum[0];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
