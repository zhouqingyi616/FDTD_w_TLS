#include <cuda_runtime.h>

#include "utils.h"

// CUDA Block size
#define BLOCK_SIZE 32

// FDTD update rule for E field: E^{n-1/2} -> E^{n+1/2}
__global__ void updateE(float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, 
    float *Cax, float *Cbx, float *Cay, float *Cby, float *Caz, float *Cbz, 
    float *Jx, float *Jy, float *Jz, float dx, int Nx, int Ny, int Nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1) 
    {
        // Iterate over z direction
        for (int k = 1; k < Nz - 1; ++k) 
        {
            int idx = i + j * Nx + k * (Nx * Ny);

            Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] * 
                      ((Hz[idx] - Hz[idx - Nx]) / dx - (Hy[idx] - Hy[idx - Nx * Ny]) / dx - Jx[idx]);
                      
            Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] * 
                      ((Hx[idx] - Hx[idx - Nx * Ny]) / dx - (Hz[idx] - Hz[idx - 1]) / dx - Jy[idx]);

            Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] * 
                      ((Hy[idx] - Hy[idx - 1]) / dx - (Hx[idx] - Hx[idx - Nx]) / dx - Jz[idx]);
        }
    }
}

// FDTD update rule for H field: H^{n} -> H^{n+1}
__global__ void updateH(float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, 
    float *Dax, float *Dbx, float *Day, float *Dby, float *Daz, float *Dbz, 
    float *Mx, float *My, float *Mz, float dx, int Nx, int Ny, int Nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1) 
    {
        // Iterate over z direction
        for (int k = 1; k < Nz - 1; ++k)
        {
            int idx = i + j * Nx + k * (Nx * Ny);
            
            Hx[idx] = Dax[idx] * Hx[idx] + Dbx[idx] * 
                      ((Ey[idx + Nx * Ny] - Ey[idx]) / dx - (Ez[idx + Nx] - Ez[idx]) / dx - Mx[idx]);
            
            Hy[idx] = Day[idx] * Hy[idx] + Dby[idx] * 
                      ((Ez[idx + 1] - Ez[idx]) / dx - (Ex[idx + Nx * Ny] - Ex[idx]) / dx - My[idx]);

            Hz[idx] = Daz[idx] * Hz[idx] + Dbz[idx] * 
                      ((Ex[idx + Nx] - Ex[idx]) / dx - (Ey[idx + 1] - Ey[idx]) / dx - Mz[idx]);
        }
    }
}

// Calculate M source based on E_rad field, for 2 faces i=i1 and i=i2
__global__ void set_M_rad_x_dir(float *Ex_rad, float *Ey_rad, float *Ez_rad, 
    float *Mx, float *My, float *Mz, float dx, int Nx, int Ny, int Nz, int N_rad, 
    int i1, int i2, int j1, int j2, int k1, int k2, int i1_rad, int i2_rad, 
    int j1_rad, int j2_rad, int k1_rad, int k2_rad)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + j1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k1;

    // Face 1: i = i1
    // My[i1-1, j, k] += -Ez_rad[i1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
    if (j <= j2 && k <= k2-1)
    {
        int idx_My = (i1 - 1) + j * Nx + k * (Nx * Ny);  // Index for My
        int idx_Ez = i1_rad + (j - j1 + j1_rad) * N_rad + (k - k1 + k1_rad) * (N_rad * N_rad);
        
        atomicAdd(&My[idx_My], -Ez_rad[idx_Ez] / dx);
    }
    // Mz[i1-1, j, k] += Ey_rad[i1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
    if (j <= j2-1 && k <= k2)
    {
        int idx_Mz = (i1 - 1) + j * Nx + k * (Nx * Ny);  // Index for My
        int idx_Ey = i1_rad + (j - j1 + j1_rad) * N_rad + (k - k1 + k1_rad) * (N_rad * N_rad);
        
        atomicAdd(&Mz[idx_Mz], Ey_rad[idx_Ey] / dx);
    }

    // Face 2: i = i2
    // My[i2, j, k] += Ez_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
    if (j <= j2 && k <= k2-1)
    {
        int idx_My = i2 + j * Nx + k * (Nx * Ny);
        int idx_Ez = i2_rad + (j - j1 + j1_rad) * N_rad + (k - k1 + k1_rad) * (N_rad * N_rad);

        atomicAdd(&My[idx_My], Ez_rad[idx_Ez] / dx);
    }
    // Mz[i2, j, k] += -Ey_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
    if (j <= j2-1 && k <= k2)
    {
        int idx_Mz = i2 + j * Nx + k * (Nx * Ny);
        int idx_Ey = i2_rad + (j - j1 + j1_rad) * N_rad + (k - k1 + k1_rad) * (N_rad * N_rad);

        atomicAdd(&Mz[idx_Mz], -Ey_rad[idx_Ey] / dx);
    }
}

// Calculate M source based on E_rad field, for 2 faces j=j1 and j=j2
__global__ void set_M_rad_y_dir(float *Ex_rad, float *Ey_rad, float *Ez_rad, 
    float *Mx, float *My, float *Mz, float dx, int Nx, int Ny, int Nz, int N_rad, 
    int i1, int i2, int j1, int j2, int k1, int k2, int i1_rad, int i2_rad, 
    int j1_rad, int j2_rad, int k1_rad, int k2_rad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + i1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k1;

    // Face 3: j = j1
    // Mx[i, j1-1, k] += Ez_rad[i, j1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
    // Mz[i, j1-1, k] += -Ex_rad[i, j1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
    if (i <= i2 && k <= k2) 
    {
        if (k <= k2 - 1) // Mx update for Face 3
        {  
            int idx_Mx = i + (j1 - 1) * Nx + k * (Nx * Ny);
            int idx_Ez = (i - i1 + i1_rad) + j1_rad * N_rad + (k - k1 + k1_rad) * (N_rad * N_rad);
            atomicAdd(&Mx[idx_Mx], Ez_rad[idx_Ez] / dx);
        }
        if (i <= i2 - 1) // Mz update for Face 3
        {  
            int idx_Mz = i + (j1 - 1) * Nx + k * (Nx * Ny);
            int idx_Ex = (i - i1 + i1_rad) + j1_rad * N_rad + (k - k1 + k1_rad) * (N_rad * N_rad);
            atomicAdd(&Mz[idx_Mz], -Ex_rad[idx_Ex] / dx);
        }
    }

    // Face 4: j = j2
    // Mx[i, j2, k] += -Ez_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
    // Mz[i, j2, k] += Ex_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
    if (i <= i2 && k <= k2) 
    {
        if (k <= k2 - 1) // Mx update for Face 4
        {  
            int idx_Mx = i + j2 * Nx + k * (Nx * Ny);
            int idx_Ez = (i - i1 + i1_rad) + j2_rad * N_rad + (k - k1 + k1_rad) * (N_rad * N_rad);
            atomicAdd(&Mx[idx_Mx], -Ez_rad[idx_Ez] / dx);
        }
        if (i <= i2 - 1) // Mz update for Face 4
        {  
            int idx_Mz = i + j2 * Nx + k * (Nx * Ny);
            int idx_Ex = (i - i1 + i1_rad) + j2_rad * N_rad + (k - k1 + k1_rad) * (N_rad * N_rad);
            atomicAdd(&Mz[idx_Mz], Ex_rad[idx_Ex] / dx);
        }
    }
}

// Calculate M source based on E_rad field, for 2 faces k=k1 and k=k2
__global__ void set_M_rad_z_dir(float *Ex_rad, float *Ey_rad, float *Ez_rad, 
    float *Mx, float *My, float *Mz, float dx, int Nx, int Ny, int Nz, int N_rad, 
    int i1, int i2, int j1, int j2, int k1, int k2, int i1_rad, int i2_rad, 
    int j1_rad, int j2_rad, int k1_rad, int k2_rad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + i1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + j1;

    // Face 5: k = k1
    // Mx[i, j, k1-1] += -Ey_rad[i, j, k1] / dx for i in (i1, i2) and j in (j1, j2-1)
    // My[i, j, k1-1] += Ex_rad[i, j, k1] / dx for i in (i1, i2-1) and j in (j1, j2)
    if (i <= i2 && j <= j2) 
    {
        if (j <= j2 - 1) // Mx update for Face 5
        {  
            int idx_Mx = i + j * Nx + (k1 - 1) * (Nx * Ny);
            int idx_Ey = (i - i1 + i1_rad) + (j - j1 + j1_rad) * N_rad + k1_rad * (N_rad * N_rad);
            atomicAdd(&Mx[idx_Mx], -Ey_rad[idx_Ey] / dx);
        }
        if (i <= i2 - 1) // My update for Face 5
        {  
            int idx_My = i + j * Nx + (k1 - 1) * (Nx * Ny);
            int idx_Ex = (i - i1 + i1_rad) + (j - j1 + j1_rad) * N_rad + k1_rad * (N_rad * N_rad);
            atomicAdd(&My[idx_My], Ex_rad[idx_Ex] / dx);
        }
    }

    // Face 6: k = k2
    // Mx[i, j, k2] += Ey_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
    // My[i, j, k2] += -Ex_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
    if (i <= i2 && j <= j2) 
    {
        if (j <= j2 - 1) // Mx update for Face 6
        {  
            int idx_Mx = i + j * Nx + k2 * (Nx * Ny);
            int idx_Ey = (i - i1 + i1_rad) + (j - j1 + j1_rad) * N_rad + k2_rad * (N_rad * N_rad);
            atomicAdd(&Mx[idx_Mx], Ey_rad[idx_Ey] / dx);
        }
        if (i <= i2 - 1) // My update for Face 6
        {  
            int idx_My = i + j * Nx + k2 * (Nx * Ny);
            int idx_Ex = (i - i1 + i1_rad) + (j - j1 + j1_rad) * N_rad + k2_rad * (N_rad * N_rad);
            atomicAdd(&My[idx_My], -Ex_rad[idx_Ex] / dx);
        }
    }
}

// Calculate J source based on H_rad field, for 2 faces i=i1 and i=i2
__global__ void set_J_rad_x_dir(float *Hx_rad, float *Hy_rad, float *Hz_rad, 
                                float *Jx, float *Jy, float *Jz, 
                                float dx, int Nx, int Ny, int Nz, int N_rad, 
                                int i1, int i2, int j1, int j2, int k1, int k2, 
                                int i1_rad, int i2_rad, int j1_rad, int j2_rad, int k1_rad, int k2_rad)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + j1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k1;

    int j_rad = j - j1 + j1_rad;
    int k_rad = k - k1 + k1_rad;

    // Face 1: i = i1
    // Jy[i1, j, k] += Hz_rad[i1-1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
    // Jz[i1, j, k] += -Hy_rad[i1-1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
    if (j <= j2 && k <= k2)
    {
        if (j <= j2 - 1)  // Jy update for Face 1
        {
            int idx_Jy = i1 + j * Nx + k * (Nx * Ny);
            int idx_Hz = (i1_rad - 1) + j_rad * N_rad + k_rad * (N_rad * N_rad);
            atomicAdd(&Jy[idx_Jy], Hz_rad[idx_Hz] / dx);
        }
        if (k <= k2 - 1)  // Jz update for Face 1
        {
            int idx_Jz = i1 + j * Nx + k * (Nx * Ny);
            int idx_Hy = (i1_rad - 1) + j_rad * N_rad + k_rad * (N_rad * N_rad);
            atomicAdd(&Jz[idx_Jz], -Hy_rad[idx_Hy] / dx);
        }
    }

    // Face 2: i = i2
    // Jy[i2, j, k] += -Hz_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
    // Jz[i2, j, k] += Hy_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
    if (j <= j2 && k <= k2)
    {
        if (j <= j2 - 1)  // Jy update for Face 2
        {
            int idx_Jy = i2 + j * Nx + k * (Nx * Ny);
            int idx_Hz = i2_rad + j_rad * N_rad + k_rad * (N_rad * N_rad);
            atomicAdd(&Jy[idx_Jy], -Hz_rad[idx_Hz] / dx);
        }
        if (k <= k2 - 1)  // Jz update for Face 2
        {
            int idx_Jz = i2 + j * Nx + k * (Nx * Ny);
            int idx_Hy = i2_rad + j_rad * N_rad + k_rad * (N_rad * N_rad);
            atomicAdd(&Jz[idx_Jz], Hy_rad[idx_Hy] / dx);
        }
    }
}

// Calculate J source based on H_rad field, for 2 faces j=j1 and j=j2
__global__ void set_J_rad_y_dir(float *Hx_rad, float *Hy_rad, float *Hz_rad, 
                                float *Jx, float *Jy, float *Jz, 
                                float dx, int Nx, int Ny, int Nz, int N_rad, 
                                int i1, int i2, int j1, int j2, int k1, int k2, 
                                int i1_rad, int i2_rad, int j1_rad, int j2_rad, int k1_rad, int k2_rad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + i1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k1;

    int i_rad = i - i1 + i1_rad;
    int k_rad = k - k1 + k1_rad;

    // Face 3: j = j1
    // Jx[i, j1, k] += -Hz_rad[i, j1-1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
    // Jz[i, j1, k] += Hx_rad[i, j1-1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
    if (i <= i2 && k <= k2)
    {
        if (i <= i2 - 1) // Jx update for Face 3
        {
            int idx_Jx = i + j1 * Nx + k * (Nx * Ny);
            int idx_Hz = i_rad + (j1_rad - 1) * N_rad + k_rad * (N_rad * N_rad);
            atomicAdd(&Jx[idx_Jx], -Hz_rad[idx_Hz] / dx);
        }
        if (k <= k2 - 1) // Jz update for Face 3
        {
            int idx_Jz = i + j1 * Nx + k * (Nx * Ny);
            int idx_Hx = i_rad + (j1_rad - 1) * N_rad + k_rad * (N_rad * N_rad);
            atomicAdd(&Jz[idx_Jz], Hx_rad[idx_Hx] / dx);
        }
    }

    // Face 4: j = j2
    // Jx[i, j2, k] += Hz_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
    // Jz[i, j2, k] += -Hx_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
    if (i <= i2 && k <= k2)
    {
        if (i <= i2 - 1) // Jx update for Face 4
        {
            int idx_Jx = i + j2 * Nx + k * (Nx * Ny);
            int idx_Hz = i_rad + j2_rad * N_rad + k_rad * (N_rad * N_rad);
            atomicAdd(&Jx[idx_Jx], Hz_rad[idx_Hz] / dx);
        }
        if (k <= k2 - 1) // Jz update for Face 4
        {
            int idx_Jz = i + j2 * Nx + k * (Nx * Ny);
            int idx_Hx = i_rad + j2_rad * N_rad + k_rad * (N_rad * N_rad);
            atomicAdd(&Jz[idx_Jz], -Hx_rad[idx_Hx] / dx);
        }
    }
}

// Calculate J source based on H_rad field, for 2 faces k=k1 and k=k2
__global__ void set_J_rad_z_dir(float *Hx_rad, float *Hy_rad, float *Hz_rad, 
                                float *Jx, float *Jy, float *Jz, 
                                float dx, int Nx, int Ny, int Nz, int N_rad, 
                                int i1, int i2, int j1, int j2, int k1, int k2, 
                                int i1_rad, int i2_rad, int j1_rad, int j2_rad, int k1_rad, int k2_rad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + i1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + j1;

    int i_rad = i - i1 + i1_rad;
    int j_rad = j - j1 + j1_rad;

    // Face 5: k = k1
    // Jx[i, j, k1] += Hy_rad[i, j, k1-1] / dx for i in (i1, i2-1) and j in (j1, j2)
    // Jy[i, j, k1] += -Hx_rad[i, j, k1-1] / dx for i in (i1, i2) and j in (j1, j2-1)
    if (i <= i2 && j <= j2)
    {
        if (i <= i2 - 1) // Jx update for Face 5
        {
            int idx_Jx = i + j * Nx + k1 * (Nx * Ny);
            int idx_Hy = i_rad + j_rad * N_rad + (k1_rad - 1) * (N_rad * N_rad);
            atomicAdd(&Jx[idx_Jx], Hy_rad[idx_Hy] / dx);
        }
        if (j <= j2 - 1) // Jy update for Face 5
        {
            int idx_Jy = i + j * Nx + k1 * (Nx * Ny);
            int idx_Hx = i_rad + j_rad * N_rad + (k1_rad - 1) * (N_rad * N_rad);
            atomicAdd(&Jy[idx_Jy], -Hx_rad[idx_Hx] / dx);
        }
    }

    // Face 6: k = k2
    // Jx[i, j, k2] += -Hy_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
    // Jy[i, j, k2] += Hx_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
    if (i <= i2 && j <= j2)
    {
        if (i <= i2 - 1) // Jx update for Face 6
        {
            int idx_Jx = i + j * Nx + k2 * (Nx * Ny);
            int idx_Hy = i_rad + j_rad * N_rad + k2_rad * (N_rad * N_rad);
            atomicAdd(&Jx[idx_Jx], -Hy_rad[idx_Hy] / dx);
        }
        if (j <= j2 - 1) // Jy update for Face 6
        {
            int idx_Jy = i + j * Nx + k2 * (Nx * Ny);
            int idx_Hx = i_rad + j_rad * N_rad + k2_rad * (N_rad * N_rad);
            atomicAdd(&Jy[idx_Jy], Hx_rad[idx_Hx] / dx);
        }
    }
}

// Utility function for allocating memory and checking for errors
void cuda_malloc_check(void **ptr, size_t size) 
{
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);  // Or handle the error as appropriate
    }
}

// Utility function for collecting monitor data E1
__global__ void save_E1_slices(float *Ex, float *Ey, float *Ez, float *d_monitor, 
    int i0, int i1, int j0, int j1, int k0, int k1, int Nx, int Ny, int save_len) 
{
    // x direction: i=i0, i1
    int j = blockIdx.x * blockDim.x + threadIdx.x + j0;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    // if (j <= j1 && k <= k1)
    if (j < j1 && k < k1)
    {
        // Face 1: i=i0, -x direction, Ez
        int idx_in_full = i0 + j * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Ez[idx_in_full];

        // Face 2: i=i1, +x direction, Ey
        idx_in_full = i1 + j * Nx + k * Nx * Ny;  // index in the full array
        idx_in_sub = save_len + (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Ey[idx_in_full];
    }

    // y direction: j=j0, j1
    int i = blockIdx.x * blockDim.x + threadIdx.x + i0;
    k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    // if (i <= i1 && k <= k1)
    if (i < i1 && k < k1)
    {
        // Face 3: j=j0, -y direction, Ex
        int idx_in_full = i + j0 * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = 2 * save_len + (i - i0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Ex[idx_in_full];

        // Face 4: j=j1, +y direction, Ez
        idx_in_full = i + j1 * Nx + k * Nx * Ny;  // index in the full array
        idx_in_sub = 3 * save_len + (i - i0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Ez[idx_in_full];
    }

    // z direction: k=k0, k1
    i = blockIdx.x * blockDim.x + threadIdx.x + i0;
    j = blockIdx.y * blockDim.y + threadIdx.y + j0;

    // if (i <= i1 && j <= j1)
    if (i < i1 && j < j1)
    {
        // Face 5: k=k0, -z direction, Ey
        int idx_in_full = i + j * Nx + k0 * Nx * Ny;  // index in the full array
        int idx_in_sub = 4 * save_len + (i - i0) * (j1 - j0 + 1) + (j - j0); // index in the subregion array
        d_monitor[idx_in_sub] = Ey[idx_in_full];
        
        // Face 6: k=k1, +z direction, Ex
        idx_in_full = i + j * Nx + k1 * Nx * Ny;  // index in the full array
        idx_in_sub = 5 * save_len + (i - i0) * (j1 - j0 + 1) + (j - j0); // index in the subregion array
        d_monitor[idx_in_sub] = Ex[idx_in_full];
    }
}

// Utility function for collecting monitor data H1
__global__ void save_H1_slices(float *Hx, float *Hy, float *Hz, float *d_monitor, 
    int i0, int i1, int j0, int j1, int k0, int k1, int Nx, int Ny, int save_len) 
{
    // x direction: i=i0, i1
    int j = blockIdx.x * blockDim.x + threadIdx.x + j0;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    if (j <= j1 && k <= k1)
    {
        // Face 1: i=i0, -x direction, Hy
        int idx_in_full = i0 + j * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Hy[idx_in_full];

        // Face 2: i=i1, +x direction, Hz
        idx_in_full = i1 + j * Nx + k * Nx * Ny;  // index in the full array
        idx_in_sub = save_len + (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Hz[idx_in_full];
    }

    // y direction: j=j0, j1
    int i = blockIdx.x * blockDim.x + threadIdx.x + i0;
    k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    if (i <= i1 && k <= k1)
    {
        // Face 3: j=j0, -y direction, Hz
        int idx_in_full = i + j0 * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = 2 * save_len + (i - i0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Hz[idx_in_full];

        // Face 4: j=j1, +y direction, Hx
        idx_in_full = i + j1 * Nx + k * Nx * Ny;  // index in the full array
        idx_in_sub = 3 * save_len + (i - i0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Hx[idx_in_full];
    }

    // z direction: k=k0, k1
    i = blockIdx.x * blockDim.x + threadIdx.x + i0;
    j = blockIdx.y * blockDim.y + threadIdx.y + j0;

    if (i <= i1 && j <= j1)
    {
        // Face 5: k=k0, -z direction, Hx
        int idx_in_full = i + j * Nx + k0 * Nx * Ny;  // index in the full array
        int idx_in_sub = 4 * save_len + (i - i0) * (j1 - j0 + 1) + (j - j0); // index in the subregion array
        d_monitor[idx_in_sub] = Hx[idx_in_full];
        
        // Face 6: k=k1, +z direction, Hy
        idx_in_full = i + j * Nx + k1 * Nx * Ny;  // index in the full array
        idx_in_sub = 5 * save_len + (i - i0) * (j1 - j0 + 1) + (j - j0); // index in the subregion array
        d_monitor[idx_in_sub] = Hy[idx_in_full];
    }
}

// Utility function for collecting monitor data E2
__global__ void save_E2_slices(float *Ex, float *Ey, float *Ez, float *d_monitor, 
    int i0, int i1, int j0, int j1, int k0, int k1, int Nx, int Ny, int save_len) 
{
    // x direction: i=i0, i1
    int j = blockIdx.x * blockDim.x + threadIdx.x + j0;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    // if (j <= j1 && k <= k1)
    if (j < j1 && k < k1)
    {
        // Face 1: i=i0, -x direction, Ey
        int idx_in_full = i0 + j * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Ey[idx_in_full];

        // Face 2: i=i1, +x direction, Ez
        idx_in_full = i1 + j * Nx + k * Nx * Ny;  // index in the full array
        idx_in_sub = save_len + (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Ez[idx_in_full];
    }

    // y direction: j=j0, j1
    int i = blockIdx.x * blockDim.x + threadIdx.x + i0;
    k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    // if (i <= i1 && k <= k1)
    if (i < i1 && k < k1)
    {
        // Face 3: j=j0, -y direction, Ez
        int idx_in_full = i + j0 * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = 2 * save_len + (i - i0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Ez[idx_in_full];

        // Face 4: j=j1, +y direction, Ex
        idx_in_full = i + j1 * Nx + k * Nx * Ny;  // index in the full array
        idx_in_sub = 3 * save_len + (i - i0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Ex[idx_in_full];
    }

    // z direction: k=k0, k1
    i = blockIdx.x * blockDim.x + threadIdx.x + i0;
    j = blockIdx.y * blockDim.y + threadIdx.y + j0;

    if (i <= i1 && j <= j1)
    // if (i < i1 && j < j1)
    {
        // Face 5: k=k0, -z direction, Ex
        int idx_in_full = i + j * Nx + k0 * Nx * Ny;  // index in the full array
        int idx_in_sub = 4 * save_len + (i - i0) * (j1 - j0 + 1) + (j - j0); // index in the subregion array
        d_monitor[idx_in_sub] = Ex[idx_in_full];
        
        // Face 6: k=k1, +z direction, Ey
        idx_in_full = i + j * Nx + k1 * Nx * Ny;  // index in the full array
        idx_in_sub = 5 * save_len + (i - i0) * (j1 - j0 + 1) + (j - j0); // index in the subregion array
        d_monitor[idx_in_sub] = Ey[idx_in_full];
    }
}

// Utility function for collecting monitor data H2
__global__ void save_H2_slices(float *Hx, float *Hy, float *Hz, float *d_monitor, 
    int i0, int i1, int j0, int j1, int k0, int k1, int Nx, int Ny, int save_len) 
{
    // x direction: i=i0, i1
    int j = blockIdx.x * blockDim.x + threadIdx.x + j0;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    if (j <= j1 && k <= k1)
    {
        // Face 1: i=i0, -x direction, Hz
        int idx_in_full = i0 + j * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Hz[idx_in_full];

        // Face 2: i=i1, +x direction, Hy
        idx_in_full = i1 + j * Nx + k * Nx * Ny;  // index in the full array
        idx_in_sub = save_len + (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Hy[idx_in_full];
    }

    // y direction: j=j0, j1
    int i = blockIdx.x * blockDim.x + threadIdx.x + i0;
    k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    if (i <= i1 && k <= k1)
    {
        // Face 3: j=j0, -y direction, Hx
        int idx_in_full = i + j0 * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = 2 * save_len + (i - i0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Hx[idx_in_full];

        // Face 4: j=j1, +y direction, Hz
        idx_in_full = i + j1 * Nx + k * Nx * Ny;  // index in the full array
        idx_in_sub = 3 * save_len + (i - i0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = Hz[idx_in_full];
    }

    // z direction: k=k0, k1
    i = blockIdx.x * blockDim.x + threadIdx.x + i0;
    j = blockIdx.y * blockDim.y + threadIdx.y + j0;

    if (i <= i1 && j <= j1)
    {
        // Face 5: k=k0, -z direction, Hy
        int idx_in_full = i + j * Nx + k0 * Nx * Ny;  // index in the full array
        int idx_in_sub = 4 * save_len + (i - i0) * (j1 - j0 + 1) + (j - j0); // index in the subregion array
        d_monitor[idx_in_sub] = Hy[idx_in_full];
        
        // Face 6: k=k1, +z direction, Hx
        idx_in_full = i + j * Nx + k1 * Nx * Ny;  // index in the full array
        idx_in_sub = 5 * save_len + (i - i0) * (j1 - j0 + 1) + (j - j0); // index in the subregion array
        d_monitor[idx_in_sub] = Hx[idx_in_full];
    }
}

// Test Maxwell-Bloch equations
void one_TLS_scatter_wo_box(bool use_correct_Gamma, bool use_correct_field)
{
    // Constants
    const float um = 1.0f;

    // Wavelength basics
    float SOURCE_FREQUENCY = 1.0f;  // Frequency of the source
    float SOURCE_WAVELENGTH = (c0 / SOURCE_FREQUENCY);
    float SOURCE_OMEGA = 2 * PI * SOURCE_FREQUENCY;

    Complex eps_air = Complex(1.0f, 0.0f);

    // Parameters: TLS
    float d0 = 2e-2;  // Dipole moment
    float omega_TLS = 1.0 * SOURCE_OMEGA;  // TLS's resonance frequency
    // Frequency will shift due to self-interaction
    
    // Decay rate inside vacuum
    float Gamma0 = pow(d0, 2.0) * pow(omega_TLS, 3.0) / (3.0f * PI * hbar * eps0 * pow(c0, 3.0));
    printf("TLS decay rate: %.7f \n", Gamma0);

    // Grid resolution dx
    float dx = 0.05f;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    float T = 100;  // Simulation time
    int max_iter = ceil(T / dt);  // Number of time-steps
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.0f;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);

    float Lx = 5 * um;
    float Ly = 5 * um;
    float Lz = 4 * um;
    int Nx = (int)ceil(Lx / dx) + 1;
    int Ny = (int)ceil(Ly / dx) + 1;
    int Nz = (int)ceil(Lz / dx) + 1;

    if (Nx % BLOCK_SIZE != 0)
        Nx = ((Nx / BLOCK_SIZE) + 1) * BLOCK_SIZE;
    if (Ny % BLOCK_SIZE != 0)
        Ny = ((Ny / BLOCK_SIZE) + 1) * BLOCK_SIZE;
    if (Nz % 2 != 0)
        Nz = ((Nz / 2) + 1) * 2;
    printf("Domain size: (%d, %d, %d) \n", Nx, Ny, Nz);
    printf("Number of grid points: %.3f Million \n", Nx * Ny * Nz / 1.0e6);
    
    // Middle of the domain
    int i_mid = Nx / 2;
    int j_mid = Ny / 2;
    int k_mid = Nz / 2;

    // Update the domain size
    Lx = (Nx - 1) * dx;
    Ly = (Ny - 1) * dx;
    Lz = (Nz - 1) * dx;

    // Total size of 3D array
    size_t size = Nx * Ny * Nz * sizeof(float);

    // Now prepare for solving Maxwell's equations

    // Initialize arrays on device
    float *Ex, *Ey, *Ez;                      // On device
    float *Hx, *Hy, *Hz;                      // On device
    float *Jx, *Jy, *Jz;                      // On device
    float *Mx, *My, *Mz;                      // On device
    float *Cax, *Cbx, *Cay, *Cby, *Caz, *Cbz; // On device
    float *Dax, *Dbx, *Day, *Dby, *Daz, *Dbz; // On device

    // Allocate memory for the E and H fields on the device
    cuda_malloc_check((void **)&Ex, size);
    cuda_malloc_check((void **)&Ey, size);
    cuda_malloc_check((void **)&Ez, size);

    cuda_malloc_check((void **)&Hx, size);
    cuda_malloc_check((void **)&Hy, size);
    cuda_malloc_check((void **)&Hz, size);

    // Allocate memory for the J and M sources on the device
    cuda_malloc_check((void **)&Jx, size);
    cuda_malloc_check((void **)&Jy, size);
    cuda_malloc_check((void **)&Jz, size);

    cuda_malloc_check((void **)&Mx, size);
    cuda_malloc_check((void **)&My, size);
    cuda_malloc_check((void **)&Mz, size);

    // Allocate memory for the C and D matrices on the device
    cuda_malloc_check((void **)&Cax, size);
    cuda_malloc_check((void **)&Cbx, size);
    cuda_malloc_check((void **)&Cay, size);
    cuda_malloc_check((void **)&Cby, size);
    cuda_malloc_check((void **)&Caz, size);
    cuda_malloc_check((void **)&Cbz, size);

    cuda_malloc_check((void **)&Dax, size);
    cuda_malloc_check((void **)&Dbx, size);
    cuda_malloc_check((void **)&Day, size);
    cuda_malloc_check((void **)&Dby, size);
    cuda_malloc_check((void **)&Daz, size);
    cuda_malloc_check((void **)&Dbz, size);

    // Initialize fields & current sources with all zeros
    cudaMemset(Ex, 0, size);
    cudaMemset(Ey, 0, size);
    cudaMemset(Ez, 0, size);

    cudaMemset(Hx, 0, size);
    cudaMemset(Hy, 0, size);
    cudaMemset(Hz, 0, size);

    cudaMemset(Jx, 0, size);
    cudaMemset(Jy, 0, size);
    cudaMemset(Jz, 0, size);

    cudaMemset(Mx, 0, size);
    cudaMemset(My, 0, size);
    cudaMemset(Mz, 0, size);

    // Initialize arrays on host
    float *Cax_host, *Cbx_host, *Cay_host, *Cby_host, *Caz_host, *Cbz_host; // On host
    float *Dax_host, *Dbx_host, *Day_host, *Dby_host, *Daz_host, *Dbz_host; // On host

    // Allocate memory for the C matrices on the host
    Cax_host = (float *)malloc(size);
    Cbx_host = (float *)malloc(size);
    Cay_host = (float *)malloc(size);
    Cby_host = (float *)malloc(size);
    Caz_host = (float *)malloc(size);
    Cbz_host = (float *)malloc(size);

    // Allocate memory for the D matrices on the host
    Dax_host = (float *)malloc(size);
    Dbx_host = (float *)malloc(size);
    Day_host = (float *)malloc(size);
    Dby_host = (float *)malloc(size);
    Daz_host = (float *)malloc(size);
    Dbz_host = (float *)malloc(size);

    // Set permittivity, permeability, conductivity (include PML)
    // Calculate Ca, Cb, Da, Db matrices.
    printf("Pre-process: setting permittivity...\n");
    set_FDTD_matrices_3D(Cax_host, Cbx_host, Cay_host, Cby_host, Caz_host, Cbz_host, 
        Dax_host, Dbx_host, Day_host, Dby_host, Daz_host, Dbz_host, 
        Nx, Ny, Nz, dx, dt, eps_air, SOURCE_OMEGA, t_PML);

    // Copy C & D matrices to device
    cudaMemcpy(Cax, Cax_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Cbx, Cbx_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Cay, Cay_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Cby, Cby_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Caz, Caz_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Cbz, Cbz_host, size, cudaMemcpyHostToDevice);

    cudaMemcpy(Dax, Dax_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Dbx, Dbx_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Day, Day_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Dby, Dby_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Daz, Daz_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Dbz, Dbz_host, size, cudaMemcpyHostToDevice);

    // Free these arrays on host, since we do not need them.
    free(Cax_host);
    free(Cbx_host);
    free(Cay_host);
    free(Cby_host);
    free(Caz_host);
    free(Cbz_host);

    free(Dax_host);
    free(Dbx_host);
    free(Day_host);
    free(Dby_host);
    free(Daz_host);
    free(Dbz_host);

    // Allocate memory for field monitors
    float *E_monitor_xy;
    E_monitor_xy = (float *)malloc(Nx * Ny * sizeof(float));

    // We also need a monitor to measure output power
    int monitor_size_half = 5;
    int monitor_size = 2 * monitor_size_half + 1;
    // x direction
    int i_power_start = i_mid - monitor_size_half;
    int i_power_end = i_mid + monitor_size_half;

    // y direction
    int j_power_start = j_mid - monitor_size_half;
    int j_power_end = j_mid + monitor_size_half;

    // z direction
    int k_power_start = k_mid - monitor_size_half;
    int k_power_end = k_mid + monitor_size_half;

    // Prepare for saving fields
    int save_len = monitor_size * monitor_size * 6;
    
    // Allocate for monitor arrays that will be used for post-processing
    float *E1_monitor = (float *)malloc((max_iter / 2) * save_len * sizeof(float));
    float *E2_monitor = (float *)malloc((max_iter / 2) * save_len * sizeof(float));

    float *H1_monitor = (float *)malloc((max_iter / 2) * save_len * sizeof(float));
    float *H2_monitor = (float *)malloc((max_iter / 2) * save_len * sizeof(float));

    if (H1_monitor == NULL || H2_monitor == NULL || 
        E1_monitor == NULL || E2_monitor == NULL) 
    {
        fprintf(stderr, "Failed to allocate memory for monitors.\n");
        exit(EXIT_FAILURE);  // Exit program with a failure status code
    }

    // Also initialize an array on GPU to save field slices
    float *tmp_monitor_GPU; // Save fields for each iteration
    cudaMalloc(&tmp_monitor_GPU, save_len * sizeof(float));
    cudaMemset(tmp_monitor_GPU, 0, save_len * sizeof(float));

    // TLS arrangement
    // In this case we put 1 TLS at the middle
    int N_TLS = 1;  // Total number of TLSs in the array

    // Position of the TLS
    std::vector<int> idx_TLS_arr;
    idx_TLS_arr.push_back(i_mid + j_mid * Nx + k_mid * (Nx * Ny));

    std::vector<Complex> rho_ee_arr;
    rho_ee_arr.push_back(Complex(0.0, 0.0));  // Start from ground state
    std::vector<Complex> rho_eg_arr;
    rho_eg_arr.push_back(Complex(0.0, 0.0));  // (0, 0.5)

    // Source: incident field
    // Construct the source: time profile of Gaussian pulse
    float freq_sigma = 0.05 * SOURCE_FREQUENCY;
    float t_sigma = 1 / freq_sigma / (2 * PI); // Used to calculate Gaussian pulse width
    float t_peak = 5 * t_sigma;
    printf("Source total time-steps: %.2f \n", (t_peak + 5 * t_sigma) / dt);

    // Incident wave
    float y_sigma = c0 * t_sigma;
    float yc_init = -5 * y_sigma; // initial center of wave packet

    float Amp_inc_wave = 0.01f;  // 0.01

    struct PlaneWaveSource source_obj = {
        Amp_inc_wave,       // Amp_E
        y_sigma,       // sigma_y
        yc_init,       // yc
        SOURCE_OMEGA / c0,       // k0
    };
    
    // Initialize an array to save Pe value
    // Using vector of vectors for 2D array
    std::vector<std::vector<float>> Pe_save_arr(N_TLS, std::vector<float>(max_iter));
    
    std::vector<float> Ex_save_arr(max_iter);
    float E_inc = 0.0, E_drive = 0.0, E_rad = 0.0;  // Driving E field at TLS position
    
    // Visualization settings
    float plot_field_bound = 5e-5;  // Plot figure: colorbar bound
    int plot_interval = 10000;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeMonitor((j_power_end - j_power_start + blockSize.x) / blockSize.x, 
        (k_power_end - k_power_start + blockSize.y) / blockSize.y);

    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;

        // Update the position of wave packet
        source_obj.yc = yc_init + c0 * t;  // Numerical dispersion?
        
        // Update E fields
        updateE<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        // Update all TLSs
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            int idx_TLS = idx_TLS_arr[id_TLS];

            // Calculate the driving E field, prepare to update TLS
            E_inc = exp(-pow(0.0 - source_obj.yc, 2.0) / pow(source_obj.sigma_y, 2) / 2);
            E_inc = source_obj.Amp_E * E_inc * sin(source_obj.k0 * (0.0 - source_obj.yc));

            E_drive = E_inc;
            cudaMemcpy(&E_rad, Ex + idx_TLS, sizeof(float), cudaMemcpyDeviceToHost);  // E_rad

            if (!use_correct_field)
                E_drive += E_rad;  // include radiation field

            Complex tmp_rho_ee = rho_ee_arr[id_TLS];
            Complex tmp_rho_eg = rho_eg_arr[id_TLS];

            // Solve ODE and update ce & cg
            // Time interval: [t, t + dt]
            int ode_steps = 5;
            Complex rho_ee_new = tmp_rho_ee; Complex rho_eg_new = tmp_rho_eg;

            float Gamma_Bloch = 0.0;  // Gamma = Gamma0 or 0, used in Bloch equation
            if (use_correct_Gamma)
                Gamma_Bloch = Gamma0;
            for (int i_step = 0; i_step < ode_steps; ++i_step)
            {
                // Correct: RK4
                std::pair<Complex, Complex> results = RK4_Bloch_step(rho_ee_new, rho_eg_new, 
                    omega_TLS, d0, E_drive, Gamma_Bloch, dt / ode_steps);
                
                rho_ee_new = results.first;
                rho_eg_new = results.second;
            }
            
            // Record result
            Pe_save_arr[id_TLS][iter] = rho_ee_new.abs();
            Ex_save_arr[iter] = E_drive;
            
            // Update the J current sources
            // The original version
            Complex Jx_value_complex = (rho_eg_new + rho_eg_new.conj()) - 
                (tmp_rho_eg + tmp_rho_eg.conj());
            float Jx_value = Jx_value_complex.real;
            Jx_value = -Jx_value * d0 / dt / pow(dx, 3);  // negative
            // It indeed should be dx^3!

            // Copy the J current source to device
            cudaMemcpy(Jx + idx_TLS, &Jx_value, sizeof(float), cudaMemcpyHostToDevice);
            
            // Update TLS status by replacing rho_ee, rho_eg
            rho_ee_arr[id_TLS] = rho_ee_new;
            rho_eg_arr[id_TLS] = rho_eg_new;
        }
        
        // Copy the M current source to device
        
        // Update H fields
        updateH<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, 
            Mx, My, Mz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        // Field monitor: update every 2 iterations to save space
        if (iter % 2 == 0)
        {
            // Save E1 fields for 6 faces
            save_E1_slices<<<gridSizeMonitor, blockSize>>>(Ex, Ey, Ez, tmp_monitor_GPU, 
                i_power_start, i_power_end, j_power_start, j_power_end, k_power_start, k_power_end, 
                Nx, Ny, monitor_size * monitor_size);
            cudaMemcpy(E1_monitor + (iter / 2) * save_len, tmp_monitor_GPU, save_len * sizeof(float), 
                cudaMemcpyDeviceToHost);  // Copy the extracted E1 to host memory
            
            // Save H1 fields for 6 faces
            save_H1_slices<<<gridSizeMonitor, blockSize>>>(Hx, Hy, Hz, tmp_monitor_GPU, 
                i_power_start, i_power_end, j_power_start, j_power_end, k_power_start, k_power_end, 
                Nx, Ny, monitor_size * monitor_size);
            cudaMemcpy(H1_monitor + (iter / 2) * save_len, tmp_monitor_GPU, save_len * sizeof(float), 
                cudaMemcpyDeviceToHost);  // Copy the extracted H1 to host memory
            
            // Save E2 fields for 6 faces
            save_E2_slices<<<gridSizeMonitor, blockSize>>>(Ex, Ey, Ez, tmp_monitor_GPU, 
                i_power_start, i_power_end, j_power_start, j_power_end, k_power_start, k_power_end, 
                Nx, Ny, monitor_size * monitor_size);
            cudaMemcpy(E2_monitor + (iter / 2) * save_len, tmp_monitor_GPU, save_len * sizeof(float), 
                cudaMemcpyDeviceToHost);  // Copy the extracted E2 to host memory
            
            // Save H2 fields for 6 faces
            save_H2_slices<<<gridSizeMonitor, blockSize>>>(Hx, Hy, Hz, tmp_monitor_GPU, 
                i_power_start, i_power_end, j_power_start, j_power_end, k_power_start, k_power_end, 
                Nx, Ny, monitor_size * monitor_size);
            cudaMemcpy(H2_monitor + (iter / 2) * save_len, tmp_monitor_GPU, save_len * sizeof(float), 
                cudaMemcpyDeviceToHost);  // Copy the extracted H2 to host memory
        }
        // Frequency monitor: update the Fourier transform if needed
        
        // Record the E field using a monitor, once in a while
        if (iter % plot_interval == 0)
        {
            printf("Iter: %d / %d \n", iter, max_iter);
            
            // Record: E field at z=0 plane
            // Copy from GPU back to CPU
            size_t slice_pitch = Nx * sizeof(float); // The size in bytes of the 2D slice row
            int k = Nz / 2; // Assuming you want the middle slice

            for (int j = 0; j < Ny; ++j) 
            {
                float* device_ptr = Ex + j * Nx + k * Nx * Ny; // Pointer to the start of the row in the desired slice
                float* host_ptr = E_monitor_xy + j * Nx;  // Pointer to the host memory

                cudaMemcpy(host_ptr, device_ptr, slice_pitch, cudaMemcpyDeviceToHost);
            }
            
            // File name
            char field_filename[50];

            snprintf(field_filename, sizeof(field_filename), "figures/Ex_%04d.png", iter);
            save_field_png(E_monitor_xy, field_filename, Nx, Ny, plot_field_bound);
        }
    }

    end = clock();  // Stop timing, display result
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Program executed in: %.4f seconds\n", cpu_time_used);

    // Free GPU memory
    cudaFree(Ex);
    cudaFree(Ey);
    cudaFree(Ez);

    cudaFree(Hx);
    cudaFree(Hy);
    cudaFree(Hz);

    cudaFree(Jx);
    cudaFree(Jy);
    cudaFree(Jz);

    cudaFree(Mx);
    cudaFree(My);
    cudaFree(Mz);

    cudaFree(Cax);
    cudaFree(Cbx);
    cudaFree(Cay);
    cudaFree(Cby);
    cudaFree(Caz);
    cudaFree(Cbz);

    cudaFree(Dax);
    cudaFree(Dbx);
    cudaFree(Day);
    cudaFree(Dby);
    cudaFree(Daz);
    cudaFree(Dbz);

    // Save the Pe data for future analysis
    std::ofstream outFile(std::string("data/Pe.csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS) 
        {
            float tmp_Pe = Pe_save_arr[id_TLS][iter];
            outFile << tmp_Pe;
            if (id_TLS < N_TLS - 1)
                outFile << ", ";  // Separate using comma
        }
        outFile << std::endl;
    }
    outFile.close();

    // Save E sequence for future analysis
    outFile = std::ofstream(std::string("data/E_drive.csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        float tmp_E_drive = Ex_save_arr[iter];
        outFile << tmp_E_drive << std::endl;
    }
    outFile.close();

    // Save the time monitor data for future analysis
    FILE *file_E1 = fopen("data/E1_monitor.bin", "wb");  // Save E1
    if (file_E1 != NULL)
    {
        printf("Saving output E1 data: max_iter = %d, save_len = %d \n", max_iter, save_len);
        for (int i = 0; i < max_iter / 2; ++i)
        {
            fwrite(&E1_monitor[i * save_len], sizeof(float), save_len, file_E1);
        }
        fclose(file_E1);
    }

    FILE *file_H1 = fopen("data/H1_monitor.bin", "wb");  // Save H1
    if (file_H1 != NULL)
    {
        printf("Saving output H1 data: max_iter = %d, save_len = %d \n", max_iter, save_len);
        for (int i = 0; i < max_iter / 2; ++i)
        {
            fwrite(&H1_monitor[i * save_len], sizeof(float), save_len, file_H1);
        }
        fclose(file_H1);
    }

    FILE *file_E2 = fopen("data/E2_monitor.bin", "wb");  // Save E2
    if (file_E2 != NULL)
    {
        printf("Saving output E2 data: max_iter = %d, save_len = %d \n", max_iter, save_len);
        for (int i = 0; i < max_iter / 2; ++i)
        {
            fwrite(&E2_monitor[i * save_len], sizeof(float), save_len, file_E2);
        }
        fclose(file_E2);
    }

    FILE *file_H2 = fopen("data/H2_monitor.bin", "wb");  // Save H1
    if (file_H2 != NULL)
    {
        printf("Saving output H2 data: max_iter = %d, save_len = %d \n", max_iter, save_len);
        for (int i = 0; i < max_iter / 2; ++i)
        {
            fwrite(&H2_monitor[i * save_len], sizeof(float), save_len, file_H2);
        }
        fclose(file_H2);
    }
    
    // Save the frequency monitor data for future analysis, if needed
    return;
}

int main()
{
    // 4 different scenarios in Fig. 1(a)
    bool use_nonzero_Gamma = false;
    bool exclude_radiation_field = false;

    // Photon scattering: 1 TLS inside vacuum
    one_TLS_scatter_wo_box(use_nonzero_Gamma, exclude_radiation_field);  // No box

    return 0;
}
