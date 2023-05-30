#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>



__global__ void solve_NavierStorks(int nx, int ny, int nit, double dx, double dy, double dt, double rho, double nu,double *u, double *v, double *p){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    double bji;
    double unji, unjip, unjim, unjpi, unjmi;
    double vnji, vnjip, vnjim, vnjpi, vnjmi;
    
    if (i>0 && i<nx-1 && j>0 && j<ny-1){
        unji = u[j*nx+i];
        unjip = u[j*nx+i+1];
        unjim = u[j*nx+i-1];
        unjpi = u[(j+1)*nx+i];
        unjmi = u[(j-1)*nx+i];

        vnji = v[j*nx+i];
        vnjip = v[j*nx+i+1];
        vnjim = v[j*nx+i-1];
        vnjpi = v[(j+1)*nx+i];
        vnjmi = v[(j-1)*nx+i];

        bji = rho * (1.0/dt * ((unjip-unjim)/(2 * dx) + (vnjpi-vnjmi)/(2 * dy)) -
                                    (unjip-unjim)*(unjip-unjim)/(4*dx*dx) -
                                    (vnjpi-vnjmi)*(vnjpi-vnjmi)/(4*dy*dy) - 
                                    2.0*(((unjpi-unjmi)/(2*dy))*((vnjip-vnjim)/(2 * dx))));
        
    }
    for (int it=0; it<nit; ++it) {
        if (i>0 && i<nx-1 && j>0 && j<ny-1){
            double pnjip = p[j*nx+i+1];
            double pnjim = p[j*nx+i-1];
            double pnjpi = p[(j+1)*nx+i];
            double pnjmi = p[(j-1)*nx+i];
            p[j*nx+i] = (dy*dy*(pnjip+pnjim)+dx*dx*(pnjpi+pnjmi)-bji*dx*dx*dy*dy)/(2.0*(dx*dx+dy*dy));
        }
        
        __syncthreads();
        if (i<nx && j==0){
            p[i] = p[ny+i];
            p[(ny-1)+i] = 0.0;
        }
        __syncthreads();
        if (j<ny && i==0){
            p[j*nx+nx-1] = p[j*nx+nx-2];
            p[j*nx+0] = p[j*nx+1];
        }
        __syncthreads();
    }
    if (i>0 && i<nx-1 && j>0 && j<ny-1){

        u[j*nx+i] = unji  - unji * dt / dx * (unji - unjim)
                          - unji * dt / dy * (unji - unjmi)
                          - dt / (2.0 * rho * dx) * (p[j*nx+i+1] - p[j*nx+i-1])
                          + nu * dt / (dx*dx) * (unjip - 2.0 * unji + unjim)
                          + nu * dt / (dy*dy) * (unjpi - 2.0 * unji + unjmi);
        v[j*nx+i] = vnji  - vnji * dt / dx * (vnji - vnjim)
                          - vnji * dt / dy * (vnji - vnjmi)
                          - dt / (2.0 * rho * dx) * (p[(j+1)*nx+i] - p[(j-1)*nx+i])
                          + nu * dt / (dx*dx) * (vnjip - 2.0 * vnji + vnjim)
                          + nu * dt / (dy*dy) * (vnjpi - 2.0 * vnji + vnjmi);
    }
    // __syncthreads();
    // if (j<ny && i==0){
    //     u[j*nx] = 0.0;
    //     u[j*nx+nx-1] = 0.0;
    //     v[j*nx] = 0.0;
    //     v[j*nx+nx-1] = 0.0;
    // }
    // __syncthreads();
    // if (i<nx && j==0){
    //     u[i] = 0.0;
    //     u[(ny-1)*nx+i] = 1.0;
    //     v[i] = 0.0;
    //     v[(ny-1)*nx+i] = 0.0;
    // }


}

int main(){
    int nx = 41;
    int ny = 41;
    int nt = 1000;
    int nit = 50;
    double dx = 2.0 / (nx - 1);
    double dy = 2.0 / (ny - 1);
    double dt = 0.01;
    double rho = 1.0;
    double nu = 0.02;

    const int size = nx*ny*sizeof(double);

    double *u;
    double *v;
    double *p;
    cudaMallocManaged(&u, size);
    cudaMallocManaged(&v, size);
    cudaMallocManaged(&p, size);

#pragma omp parallel for
    for (int j=0; j<ny; ++j){
        u[j*nx] = 0.0;
        u[j*nx+nx-1] = 0.0;
        v[j*nx] = 0.0;
        v[j*nx+nx-1] = 0.0;
    }

#pragma omp parallel for
    for (int i=0; i<nx; ++i){
        u[i] = 0.0;
        u[(ny-1)*nx+i] = 1.0;
        v[i] = 0.0;
        v[(ny-1)*nx+i] = 0.0;
    }
    

    dim3 block(10,10);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);

    for (int n=0;n<nt;++n){
        solve_NavierStorks<<<grid,block>>>(nx,ny,nit,dx,dy,dt,rho,nu,u,v,p);
        cudaDeviceSynchronize();
        for (int i=0; i<nx; ++i){
            for (int j=0; j<ny; ++j){
                std::cout << dt*n << " "<< dx*i << " " << dy*j << " " << u[j*nx+i] << " " << v[j*nx+i] << " " << p[j*nx+i] << std::endl;
            }
        }
    }

    cudaFree(u);
    cudaFree(v);
    cudaFree(p);

    return 0;
}

