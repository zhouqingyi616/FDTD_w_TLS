#include <cuda_runtime.h>

#include "utils.h"

// CUDA Block size
#define BLOCK_SIZE 16

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

// FDTD update rule for E field: E^{n-1/2} -> E^{n+1/2}, with PEC
__global__ void updateE_PEC(float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, 
    float *Cax, float *Cbx, float *Cay, float *Cby, float *Caz, float *Cbz, 
    float *Jx, float *Jy, float *Jz, float dx, int j_PEC, int Nx, int Ny, int Nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1) 
    {
        // Iterate over z direction
        for (int k = 1; k < Nz - 1; ++k) 
        {
            int idx = i + j * Nx + k * (Nx * Ny);

            if (j >= j_PEC)
            {
                Ex[idx] = 0.0;
                Ez[idx] = 0.0;
            }
            else
            {
                Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] * 
                    ((Hz[idx] - Hz[idx - Nx]) / dx - (Hy[idx] - Hy[idx - Nx * Ny]) / dx - Jx[idx]);
                Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] * 
                    ((Hy[idx] - Hy[idx - 1]) / dx - (Hx[idx] - Hx[idx - Nx]) / dx - Jz[idx]);
            }

            Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] * 
                      ((Hx[idx] - Hx[idx - Nx * Ny]) / dx - (Hz[idx] - Hz[idx - 1]) / dx - Jy[idx]);
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

// The most naive version: no TF-IF box
void one_TLS_decay_vacuum_wo_box()
{
    // Constants
    const float um = 1.0f;

    // Wavelength basics
    float SOURCE_FREQUENCY = 1.0f;  // Frequency of the source
    float SOURCE_WAVELENGTH = (c0 / SOURCE_FREQUENCY);
    float SOURCE_OMEGA = 2 * PI * SOURCE_FREQUENCY;

    Complex eps_air = Complex(1.0f, 0.0f);

    // Parameters: TLS
    float d0 = 1e-2;  // Dipole moment
    float omega_TLS = 1.0 * SOURCE_OMEGA;  // TLS's resonance frequency
    // Frequency will shift due to self-interaction
    
    // Decay rate inside vacuum
    float Gamma0 = pow(d0, 2.0) * pow(omega_TLS, 3.0) / (3.0f * PI * hbar * eps0 * pow(c0, 3.0));
    printf("TLS decay rate: %.7f \n", Gamma0);

    // Grid resolution dx
    float dx = 0.05f;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    // float T = 1000 / SOURCE_FREQUENCY;  // 10 * T0
    int max_iter = 5000;  // Number of time-steps
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.5f;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);

    float Lx = 8 * um;
    float Ly = 8 * um;
    float Lz = 8 * um;
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
    float *Jx_host, *Jy_host, *Jz_host;                      // On host
    float *Mx_host, *My_host, *Mz_host;                      // On host
    float *Cax_host, *Cbx_host, *Cay_host, *Cby_host, *Caz_host, *Cbz_host; // On host
    float *Dax_host, *Dbx_host, *Day_host, *Dby_host, *Daz_host, *Dbz_host; // On host

    // Allocate memory for the J and M sources on the host
    Jx_host = (float *)malloc(size);
    Jy_host = (float *)malloc(size);
    Jz_host = (float *)malloc(size);

    Mx_host = (float *)malloc(size);
    My_host = (float *)malloc(size);
    Mz_host = (float *)malloc(size);

    // Initialize current sources!
    memset(Jx_host, 0, size);
    memset(Jy_host, 0, size);
    memset(Jz_host, 0, size);
    
    memset(Mx_host, 0, size);
    memset(My_host, 0, size);
    memset(Mz_host, 0, size);

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

    // TLS arrangement
    // In this case we put 1 TLS at the middle
    int row_TLS = 1;
    int col_TLS = 1;
    int N_TLS = row_TLS * col_TLS;  // Total number of TLSs in the array

    float neighbor_dist = 5 * dx;

    float Lx_TLS_arr = (row_TLS - 1) * neighbor_dist;
    float Ly_TLS_arr = (col_TLS - 1) * neighbor_dist;

    // The center of the TLS array
    float xc_TLS = 0.0;
    float yc_TLS = 0.0;
    float zc_TLS = 0.0;

    // The corner of the TLS array
    float x_min_TLS = xc_TLS - Lx_TLS_arr / 2.0;
    float y_min_TLS = yc_TLS - Ly_TLS_arr / 2.0;
    float z_TLS = zc_TLS;

    // Position of the TLS
    std::vector<int> idx_TLS_arr;

    std::vector<Complex> b_arr;

    for (int row = 0; row < row_TLS; ++row)
    {
        for (int col = 0; col < col_TLS; ++col)
        {
            float x_TLS = x_min_TLS + row * neighbor_dist;
            float y_TLS = y_min_TLS + col * neighbor_dist;

            int i_TLS = i_mid + round(x_TLS / dx);
            int j_TLS = j_mid + round(y_TLS / dx);
            int k_TLS = k_mid + round(z_TLS / dx);

            idx_TLS_arr.push_back(i_TLS + j_TLS * Nx + k_TLS * (Nx * Ny));

            float b_init_value = 1.0 / sqrtf(N_TLS);
            
            b_arr.push_back(Complex(0, b_init_value));  // Start from ground state
            
            printf("TLS located at pixel (%d, %d, %d) \n", i_TLS, j_TLS, k_TLS);
        }
    }

    // We don't have extra current source now...
    
    // Initialize an array to save b value
    // Using vector of vectors for 2D array
    std::vector<std::vector<Complex>> b_save_arr(N_TLS, std::vector<Complex>(max_iter));
    
    std::vector<float> Ex_save_arr(max_iter);
    float tmp_E_drive = 0.0;  // To save Ez field at TLS position
    
    // Visualization settings
    float plot_field_bound = 0.05;  // Plot figure: colorbar bound
    int plot_interval = 1200;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);

    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;

        // Copy the J current source to device
        cudaMemcpy(Jx, Jx_host, size, cudaMemcpyHostToDevice);
        // cudaMemcpy(Jy, Jy_host, size, cudaMemcpyHostToDevice);
        // cudaMemcpy(Jz, Jz_host, size, cudaMemcpyHostToDevice);
        
        // Update E fields
        updateE<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        // Update all TLSs
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            int idx_TLS = idx_TLS_arr[id_TLS];

            // Copy Ez back to host, prepare to update bz
            cudaMemcpy(&tmp_E_drive, Ex + idx_TLS, sizeof(float), cudaMemcpyDeviceToHost);

            Complex tmp_b = b_arr[id_TLS];

            // Solve ODE and update b
            // Time interval: [t, t + dt]
            int ode_steps = 5;
            Complex b_new = tmp_b;
            for (int i_step = 0; i_step < ode_steps; ++i_step)
                b_new = RK4_step(t, b_new, dt / ode_steps, omega_TLS, d0, Gamma0, tmp_E_drive, false);

            Complex b_mid = (b_arr[id_TLS] + b_new) / 2.0;
            
            // Record result
            b_save_arr[id_TLS][iter] = b_new;
            Ex_save_arr[iter] = tmp_E_drive;
            
            // Update the J current sources
            // The original version
            Jx_host[idx_TLS] = +2 * d0 * omega_TLS * b_new.imag / pow(dx, 3);
            // It indeed should be dx^3!
            
            // Update TLS status by replacing b
            b_arr[id_TLS] = b_new;  // b_new
        }
        /*
        // Copy the M current source to device
        cudaMemcpy(Mx, Mx_host, size, cudaMemcpyHostToDevice);
        cudaMemcpy(My, My_host, size, cudaMemcpyHostToDevice);
        cudaMemcpy(Mz, Mz_host, size, cudaMemcpyHostToDevice);
        */
        // Update H fields
        updateH<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, 
            Mx, My, Mz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        // Field monitor: update if needed

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

    // Free CPU memory
    free(Jx_host);
    free(Jy_host);
    free(Jz_host);

    free(Mx_host);
    free(My_host);
    free(Mz_host);

    // Save the b data for future analysis
    std::ofstream outFile(std::string("data/b_N=") + std::to_string(N_TLS) + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS) 
        {
            Complex tmp_b = b_save_arr[id_TLS][iter];
            if (tmp_b.imag >= 0)
                outFile << tmp_b.real << "+" << tmp_b.imag << "j";
            else
                outFile << tmp_b.real << "-" << -tmp_b.imag << "j";
            if (id_TLS < N_TLS - 1)
                outFile << ", ";  // Separate using comma
        }
        outFile << std::endl;
    }
    outFile.close();

    // Save Ez sequence for future analysis
    outFile = std::ofstream(std::string("data/Ex") + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        float tmp_E_drive = Ex_save_arr[iter];
        outFile << tmp_E_drive << std::endl;
    }
    outFile.close();

    // Save the frequency monitor data for future analysis, if needed
    return;
}

// The modified version: with TF-IF box
void one_TLS_decay_vacuum_w_box()
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
    float dx = 0.03f;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    float T = 300;  // 10 * T0
    int max_iter = ceil(T / dt);  // Number of time-steps
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.0f;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);

    float Lx = 6 * um;
    float Ly = 6 * um;
    float Lz = 6 * um;
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

    // We need to define another domain for FDTD of radiation field
    int box_half = 1;  // Box size: 2 * box_half + 1
    int N_rad_half = box_half + t_PML + ceil(0.8 * SOURCE_WAVELENGTH / dx);  // + 1
    int N_rad = 2 * N_rad_half + 1;
    int N_rad_mid = N_rad_half + 1;  // The dipole position
    printf("Extra FDTD size: (%d, %d, %d) \n", N_rad, N_rad, N_rad);

    // Total size of 3D array
    size_t size = Nx * Ny * Nz * sizeof(float);
    size_t size_rad = N_rad * N_rad * N_rad * sizeof(float);

    // Now prepare for solving Maxwell's equations

    // Initialize arrays on device
    float *Ex, *Ey, *Ez;                      // On device
    float *Hx, *Hy, *Hz;                      // On device
    float *Jx, *Jy, *Jz;                      // On device
    float *Mx, *My, *Mz;                      // On device
    float *Cax, *Cbx, *Cay, *Cby, *Caz, *Cbz; // On device
    float *Dax, *Dbx, *Day, *Dby, *Daz, *Dbz; // On device

    // The radiation field
    float *Ex_rad, *Ey_rad, *Ez_rad;                      // On device
    float *Hx_rad, *Hy_rad, *Hz_rad;                      // On device
    float *Jx_rad, *Jy_rad, *Jz_rad;                      // On device
    float *Mx_rad, *My_rad, *Mz_rad;                      // On device
    float *Cax_rad, *Cbx_rad, *Cay_rad, *Cby_rad, *Caz_rad, *Cbz_rad; // On device
    float *Dax_rad, *Dbx_rad, *Day_rad, *Dby_rad, *Daz_rad, *Dbz_rad; // On device

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

    // Do similar things for radiation field
    // Allocate memory for the E and H fields on the device
    cuda_malloc_check((void **)&Ex_rad, size_rad);
    cuda_malloc_check((void **)&Ey_rad, size_rad);
    cuda_malloc_check((void **)&Ez_rad, size_rad);

    cuda_malloc_check((void **)&Hx_rad, size_rad);
    cuda_malloc_check((void **)&Hy_rad, size_rad);
    cuda_malloc_check((void **)&Hz_rad, size_rad);

    // Allocate memory for the J and M sources on the device
    cuda_malloc_check((void **)&Jx_rad, size_rad);
    cuda_malloc_check((void **)&Jy_rad, size_rad);
    cuda_malloc_check((void **)&Jz_rad, size_rad);

    cuda_malloc_check((void **)&Mx_rad, size_rad);
    cuda_malloc_check((void **)&My_rad, size_rad);
    cuda_malloc_check((void **)&Mz_rad, size_rad);

    // Allocate memory for the C and D matrices on the device
    cuda_malloc_check((void **)&Cax_rad, size_rad);
    cuda_malloc_check((void **)&Cbx_rad, size_rad);
    cuda_malloc_check((void **)&Cay_rad, size_rad);
    cuda_malloc_check((void **)&Cby_rad, size_rad);
    cuda_malloc_check((void **)&Caz_rad, size_rad);
    cuda_malloc_check((void **)&Cbz_rad, size_rad);

    cuda_malloc_check((void **)&Dax_rad, size_rad);
    cuda_malloc_check((void **)&Dbx_rad, size_rad);
    cuda_malloc_check((void **)&Day_rad, size_rad);
    cuda_malloc_check((void **)&Dby_rad, size_rad);
    cuda_malloc_check((void **)&Daz_rad, size_rad);
    cuda_malloc_check((void **)&Dbz_rad, size_rad);

    // Initialize fields & current sources with all zeros
    cudaMemset(Ex_rad, 0, size_rad);
    cudaMemset(Ey_rad, 0, size_rad);
    cudaMemset(Ez_rad, 0, size_rad);

    cudaMemset(Hx_rad, 0, size_rad);
    cudaMemset(Hy_rad, 0, size_rad);
    cudaMemset(Hz_rad, 0, size_rad);

    cudaMemset(Jx_rad, 0, size_rad);
    cudaMemset(Jy_rad, 0, size_rad);
    cudaMemset(Jz_rad, 0, size_rad);

    cudaMemset(Mx_rad, 0, size_rad);
    cudaMemset(My_rad, 0, size_rad);
    cudaMemset(Mz_rad, 0, size_rad);

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
    
    // Also initialize structure for radiation field FDTD
    float *Cax_rad_host, *Cbx_rad_host, *Cay_rad_host, *Cby_rad_host, *Caz_rad_host, *Cbz_rad_host; // On host
    float *Dax_rad_host, *Dbx_rad_host, *Day_rad_host, *Dby_rad_host, *Daz_rad_host, *Dbz_rad_host; // On host

    // Allocate memory for the C matrices on the host
    Cax_rad_host = (float *)malloc(size_rad);
    Cbx_rad_host = (float *)malloc(size_rad);
    Cay_rad_host = (float *)malloc(size_rad);
    Cby_rad_host = (float *)malloc(size_rad);
    Caz_rad_host = (float *)malloc(size_rad);
    Cbz_rad_host = (float *)malloc(size_rad);

    // Allocate memory for the D matrices on the host
    Dax_rad_host = (float *)malloc(size_rad);
    Dbx_rad_host = (float *)malloc(size_rad);
    Day_rad_host = (float *)malloc(size_rad);
    Dby_rad_host = (float *)malloc(size_rad);
    Daz_rad_host = (float *)malloc(size_rad);
    Dbz_rad_host = (float *)malloc(size_rad);

    set_FDTD_matrices_3D(Cax_rad_host, Cbx_rad_host, Cay_rad_host, Cby_rad_host, 
        Caz_rad_host, Cbz_rad_host, Dax_rad_host, Dbx_rad_host, Day_rad_host, 
        Dby_rad_host, Daz_rad_host, Dbz_rad_host, 
        N_rad, N_rad, N_rad, dx, dt, eps_air, SOURCE_OMEGA, t_PML);

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

    cudaMemcpy(Cax_rad, Cax_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cbx_rad, Cbx_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cay_rad, Cay_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cby_rad, Cby_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Caz_rad, Caz_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cbz_rad, Cbz_rad_host, size_rad, cudaMemcpyHostToDevice);

    cudaMemcpy(Dax_rad, Dax_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dbx_rad, Dbx_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Day_rad, Day_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dby_rad, Dby_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Daz_rad, Daz_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dbz_rad, Dbz_rad_host, size_rad, cudaMemcpyHostToDevice);

    // Free these arrays on host, since we do not need them anymore.
    free(Cax_rad_host);
    free(Cbx_rad_host);
    free(Cay_rad_host);
    free(Cby_rad_host);
    free(Caz_rad_host);
    free(Cbz_rad_host);

    free(Dax_rad_host);
    free(Dbx_rad_host);
    free(Day_rad_host);
    free(Dby_rad_host);
    free(Daz_rad_host);
    free(Dbz_rad_host);

    // Allocate memory for field monitors
    float *E_monitor_xy, *E_rad_monitor_xy;
    E_monitor_xy = (float *)malloc(Nx * Ny * sizeof(float));
    E_rad_monitor_xy = (float *)malloc(N_rad * N_rad * sizeof(float));

    // TLS arrangement
    // In this case we put 1 TLS at the middle
    int N_TLS = 1;  // Total number of TLSs

    // Position of the TLS
    int i_TLS = i_mid;
    int j_TLS = j_mid;
    int k_TLS = k_mid;
    printf("TLS located at pixel (%d, %d, %d) \n", i_TLS, j_TLS, k_TLS);

    // Index of TLS
    int idx_TLS = i_TLS + j_TLS * Nx + k_TLS * (Nx * Ny);
    int idx_TLS_rad = N_rad_mid + N_rad_mid * N_rad + N_rad_mid * (N_rad * N_rad);

    // Indices for box boundaries [i1, i2] x [j1, j2] x [k1, k2]
    int i1 = i_mid - box_half; int i2 = i_mid + box_half;
    int j1 = j_mid - box_half; int j2 = j_mid + box_half;
    int k1 = k_mid - box_half; int k2 = k_mid + box_half;

    int i1_rad = N_rad_mid - box_half; int i2_rad = N_rad_mid + box_half;
    int j1_rad = N_rad_mid - box_half; int j2_rad = N_rad_mid + box_half;
    int k1_rad = N_rad_mid - box_half; int k2_rad = N_rad_mid + box_half;

    // Initialize TLS: start from excited state
    Complex b = Complex(1.0, 0.0);

    // We don't have extra current source now...
    
    // Initialize an array to save b value
    // Using vector of vectors for 2D array
    std::vector<std::vector<Complex>> b_save_arr(N_TLS, std::vector<Complex>(max_iter));
    
    std::vector<float> Ex_save_arr(max_iter);
    float tmp_E_drive = 0.0;  // To save Ez field at TLS position
    
    // Visualization settings
    float plot_field_bound = 0.1;  // Plot figure: colorbar bound
    int plot_interval = 1000;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeRad((N_rad + blockSize.x - 1) / blockSize.x, (N_rad + blockSize.y - 1) / blockSize.y);

    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;

        // We will now carry out the 8 steps one-by-one!

        // Step 1: calculate J_{i}^{n} from b_{i}^{n}
        float Jx_value = 2 * d0 * omega_TLS * b.imag / pow(dx, 3);
        cudaMemcpy(Jx_rad + idx_TLS_rad, &Jx_value, sizeof(float), cudaMemcpyHostToDevice);

        // Step 2: update E_{i} based on J_{i}^{n}
        updateE<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, 
            Hx_rad, Hy_rad, Hz_rad, Cax_rad, Cbx_rad, Cay_rad, Cby_rad, 
            Caz_rad, Cbz_rad, Jx_rad, Jy_rad, Jz_rad, dx, N_rad, N_rad, N_rad);
        cudaDeviceSynchronize();

        // Step 3: calculate J_{i, rad}^{n} based on H_{i}^{n}
        // Equivalence principle for J
        // Set Jx, Jy, Jz
        
        // Face 1: i = i1
        // Jy[i1, j, k] += Hz_rad[i1-1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Jz[i1, j, k] += -Hy_rad[i1-1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Face 2: i = i2
        // Jy[i2, j, k] += -Hz_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Jz[i2, j, k] += Hy_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        set_J_rad_x_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 3: j = j1
        // Jx[i, j1, k] += -Hz_rad[i, j1-1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Jz[i, j1, k] += Hx_rad[i, j1-1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Face 4: j = j2
        // Jx[i, j2, k] += Hz_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Jz[i, j2, k] += -Hx_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        set_J_rad_y_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 5: k = k1
        // Jx[i, j, k1] += Hy_rad[i, j, k1-1] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Jy[i, j, k1] += -Hx_rad[i, j, k1-1] / dx for i in (i1, i2) and j in (j1, j2-1)
        // Face 6: k = k2
        // Jx[i, j, k2] += -Hy_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Jy[i, j, k2] += Hx_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
        set_J_rad_z_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Step 4: update E based on J_{i, rad}^{n}
        updateE<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();
        // Don't forget to reset the current sources to all 0's!S
        cudaMemset(Jx, 0, size);
        cudaMemset(Jy, 0, size);
        cudaMemset(Jz, 0, size);

        // Step 5: calculate M_{i, rad}^{n+1/2} based on E_{i}^{n+1/2}
        // Equivalence principle for M
        // Set Mx, My and Mz
        
        // Face 1: i = i1
        // My[i1-1, j, k] += -Ez_rad[i1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Mz[i1-1, j, k] += Ey_rad[i1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Face 2: i = i2
        // My[i2, j, k] += Ez_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Mz[i2, j, k] += -Ey_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        set_M_rad_x_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Mx[i, j1-1, k] += Ez_rad[i, j1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Mz[i, j1-1, k] += -Ex_rad[i, j1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Face 4: j = j2
        // Mx[i, j2, k] += -Ez_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Mz[i, j2, k] += Ex_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        set_M_rad_y_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 5: k = k1
        // Mx[i, j, k1-1] += -Ey_rad[i, j, k1] / dx for i in (i1, i2) and j in (j1, j2-1)
        // My[i, j, k1-1] += Ex_rad[i, j, k1] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Face 6: k = k2
        // Mx[i, j, k2] += Ey_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
        // My[i, j, k2] += -Ex_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
        set_M_rad_z_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Step 6: update H based on M_{i, rad}^{n+1/2}
        updateH<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, 
            Mx, My, Mz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();
        // Don't forget to reset the current sources to all 0's!S
        cudaMemset(Mx, 0, size);
        cudaMemset(My, 0, size);
        cudaMemset(Mz, 0, size);

        // Step 7: update H_{i} based on H_{i}^{n}
        updateH<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, 
            Hx_rad, Hy_rad, Hz_rad, Dax_rad, Dbx_rad, Day_rad, 
            Dby_rad, Daz_rad, Dbz_rad, Mx_rad, My_rad, Mz_rad, 
            dx, N_rad, N_rad, N_rad);
        cudaDeviceSynchronize();

        // Step 8: update b
        // Get the field: copy Ez back to host, prepare to update bz
        cudaMemcpy(&tmp_E_drive, Ex + idx_TLS, sizeof(float), cudaMemcpyDeviceToHost);
        // b = Complex(0, 1) * Complex(cos(omega_TLS * t), -sin(omega_TLS * t)); // perfect oscillating dipole
        
        // Solve ODE and update b
        // Time interval: [t, t + dt]
        int ode_steps = 5;
        Complex b_new = b;
        for (int i_step = 0; i_step < ode_steps; ++i_step)
            b_new = RK4_step(t, b_new, dt / ode_steps, omega_TLS, d0, Gamma0, tmp_E_drive, true);
            
        // Record result
        b_save_arr[0][iter] = b_new;
        Ex_save_arr[iter] = tmp_E_drive;
        b = b_new;  // Update
        
        // Field monitor: update if needed

        // Frequency monitor: update the Fourier transform if needed
        
        // Record the E field using a monitor, once in a while
        if (iter % plot_interval == 0)
        {
            printf("Iter: %d / %d \n", iter, max_iter);
            
            // File name initialization
            char field_filename[50];
            
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
            
            snprintf(field_filename, sizeof(field_filename), "figures/Ex_%04d.png", iter);
            save_field_png(E_monitor_xy, field_filename, Nx, Ny, plot_field_bound);

            // Record: E_rad field at z=0 plane
            size_t slice_pitch_rad = N_rad * sizeof(float); // The size in bytes of the 2D slice row
            k = N_rad_mid;  // Assuming you want the middle slice

            for (int j = 0; j < N_rad; ++j)
            {
                float* device_ptr = Ex_rad + j * N_rad + k * N_rad * N_rad; // Pointer to the start of the row in the desired slice
                float* host_ptr = E_rad_monitor_xy + j * N_rad;  // Pointer to the host memory

                cudaMemcpy(host_ptr, device_ptr, slice_pitch_rad, cudaMemcpyDeviceToHost);
            }
            
            snprintf(field_filename, sizeof(field_filename), "figures/Ex_rad_%04d.png", iter);
            save_field_png(E_rad_monitor_xy, field_filename, N_rad, N_rad, plot_field_bound);
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

    // Don't forget radiation FDTD
    cudaFree(Ex_rad);
    cudaFree(Ey_rad);
    cudaFree(Ez_rad);

    cudaFree(Hx_rad);
    cudaFree(Hy_rad);
    cudaFree(Hz_rad);

    cudaFree(Jx_rad);
    cudaFree(Jy_rad);
    cudaFree(Jz_rad);

    cudaFree(Mx_rad);
    cudaFree(My_rad);
    cudaFree(Mz_rad);

    cudaFree(Cax_rad);
    cudaFree(Cbx_rad);
    cudaFree(Cay_rad);
    cudaFree(Cby_rad);
    cudaFree(Caz_rad);
    cudaFree(Cbz_rad);

    cudaFree(Dax_rad);
    cudaFree(Dbx_rad);
    cudaFree(Day_rad);
    cudaFree(Dby_rad);
    cudaFree(Daz_rad);
    cudaFree(Dbz_rad);

    // Save the b data for future analysis
    std::ofstream outFile(std::string("data/b_N=") + std::to_string(N_TLS) + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS) 
        {
            Complex tmp_b = b_save_arr[id_TLS][iter];
            if (tmp_b.imag >= 0)
                outFile << tmp_b.real << "+" << tmp_b.imag << "j";
            else
                outFile << tmp_b.real << "-" << -tmp_b.imag << "j";
            if (id_TLS < N_TLS - 1)
                outFile << ", ";  // Separate using comma
        }
        outFile << std::endl;
    }
    outFile.close();

    // Save Ez sequence for future analysis
    outFile = std::ofstream(std::string("data/Ex") + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        float tmp_E_drive = Ex_save_arr[iter];
        outFile << tmp_E_drive << std::endl;
    }
    outFile.close();

    // Save the frequency monitor data for future analysis, if needed
    return;
}

void one_TLS_decay_PEC_wo_box()
{
    // Constants
    const float um = 1.0f;

    // Wavelength basics
    float SOURCE_FREQUENCY = 1.0f;  // Frequency of the source
    float SOURCE_WAVELENGTH = (c0 / SOURCE_FREQUENCY);
    float SOURCE_OMEGA = 2 * PI * SOURCE_FREQUENCY;

    Complex eps_air = Complex(1.0f, 0.0f);

    // Parameters: TLS
    float d0 = 1e-2;  // Dipole moment
    float omega_TLS = 1.0 * SOURCE_OMEGA;  // TLS's resonance frequency
    // Frequency will shift due to self-interaction
    
    // Decay rate inside vacuum
    float Gamma0 = pow(d0, 2.0) * pow(omega_TLS, 3.0) / (3.0f * PI * hbar * eps0 * pow(c0, 3.0));
    printf("TLS decay rate: %.7f \n", Gamma0);

    // Grid resolution dx
    float dx = 0.025f;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    // float T = 1000 / SOURCE_FREQUENCY;  // 10 * T0
    int max_iter = 20000;  // Number of time-steps
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.2f;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);

    // Distance to PEC mirror
    float h_PEC = 0.1f;
    float h_PEC_grid = round(h_PEC / dx);
    h_PEC = h_PEC_grid * dx;
    printf("Distance to mirror: %.3f um \n", h_PEC);

    float Lx = 6 * um;
    float Ly = 6 * um;
    float Lz = 6 * um;
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

    // PEC: j > j_mid + h_PEC_grid
    int j_PEC = j_mid + h_PEC_grid;

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
    float *Jx_host, *Jy_host, *Jz_host;                      // On host
    float *Mx_host, *My_host, *Mz_host;                      // On host
    float *Cax_host, *Cbx_host, *Cay_host, *Cby_host, *Caz_host, *Cbz_host; // On host
    float *Dax_host, *Dbx_host, *Day_host, *Dby_host, *Daz_host, *Dbz_host; // On host

    // Allocate memory for the J and M sources on the host
    Jx_host = (float *)malloc(size);
    Jy_host = (float *)malloc(size);
    Jz_host = (float *)malloc(size);

    Mx_host = (float *)malloc(size);
    My_host = (float *)malloc(size);
    Mz_host = (float *)malloc(size);

    // Initialize current sources!
    memset(Jx_host, 0, size);
    memset(Jy_host, 0, size);
    memset(Jz_host, 0, size);
    
    memset(Mx_host, 0, size);
    memset(My_host, 0, size);
    memset(Mz_host, 0, size);

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

    // TLS arrangement
    // In this case we put 1 TLS at the middle
    int row_TLS = 1;
    int col_TLS = 1;
    int N_TLS = row_TLS * col_TLS;  // Total number of TLSs in the array

    float neighbor_dist = 5 * dx;

    float Lx_TLS_arr = (row_TLS - 1) * neighbor_dist;
    float Ly_TLS_arr = (col_TLS - 1) * neighbor_dist;

    // The center of the TLS array
    float xc_TLS = 0.0;
    float yc_TLS = 0.0;
    float zc_TLS = 0.0;

    // The corner of the TLS array
    float x_min_TLS = xc_TLS - Lx_TLS_arr / 2.0;
    float y_min_TLS = yc_TLS - Ly_TLS_arr / 2.0;
    float z_TLS = zc_TLS;

    // Position of the TLS
    std::vector<int> idx_TLS_arr;

    std::vector<Complex> b_arr;

    for (int row = 0; row < row_TLS; ++row)
    {
        for (int col = 0; col < col_TLS; ++col)
        {
            float x_TLS = x_min_TLS + row * neighbor_dist;
            float y_TLS = y_min_TLS + col * neighbor_dist;

            int i_TLS = i_mid + round(x_TLS / dx);
            int j_TLS = j_mid + round(y_TLS / dx);
            int k_TLS = k_mid + round(z_TLS / dx);

            idx_TLS_arr.push_back(i_TLS + j_TLS * Nx + k_TLS * (Nx * Ny));

            float b_init_value = 1.0 / sqrtf(N_TLS);
            
            b_arr.push_back(Complex(0, b_init_value));  // Start from ground state
            
            printf("TLS located at pixel (%d, %d, %d) \n", i_TLS, j_TLS, k_TLS);
        }
    }

    // We don't have extra current source now...
    
    // Initialize an array to save b value
    // Using vector of vectors for 2D array
    std::vector<std::vector<Complex>> b_save_arr(N_TLS, std::vector<Complex>(max_iter));
    
    std::vector<float> Ex_save_arr(max_iter);
    float tmp_E_drive = 0.0;  // To save Ez field at TLS position
    
    // Visualization settings
    float plot_field_bound = 0.05;  // Plot figure: colorbar bound
    int plot_interval = 500;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);

    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;

        // Copy the J current source to device
        cudaMemcpy(Jx, Jx_host, size, cudaMemcpyHostToDevice);
        // cudaMemcpy(Jy, Jy_host, size, cudaMemcpyHostToDevice);
        // cudaMemcpy(Jz, Jz_host, size, cudaMemcpyHostToDevice);
        
        // Update E fields
        updateE_PEC<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, j_PEC, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        // Update all TLSs
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            int idx_TLS = idx_TLS_arr[id_TLS];

            // Copy Ez back to host, prepare to update bz
            cudaMemcpy(&tmp_E_drive, Ex + idx_TLS, sizeof(float), cudaMemcpyDeviceToHost);

            Complex tmp_b = b_arr[id_TLS];

            // Solve ODE and update b
            // Time interval: [t, t + dt]
            int ode_steps = 5;
            Complex b_new = tmp_b;
            for (int i_step = 0; i_step < ode_steps; ++i_step)
                b_new = RK4_step(t, b_new, dt / ode_steps, omega_TLS, d0, Gamma0, tmp_E_drive, false);

            Complex b_mid = (b_arr[id_TLS] + b_new) / 2.0;
            
            // Record result
            b_save_arr[id_TLS][iter] = b_new;
            Ex_save_arr[iter] = tmp_E_drive;
            
            // Update the J current sources
            // The original version
            Jx_host[idx_TLS] = +2 * d0 * omega_TLS * b_new.imag / pow(dx, 3);
            // It indeed should be dx^3!
            
            // Update TLS status by replacing b
            b_arr[id_TLS] = b_new;  // b_new
        }
        /*
        // Copy the M current source to device
        */
        // Update H fields
        updateH<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, 
            Mx, My, Mz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        // Field monitor: update if needed

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

    // Free CPU memory
    free(Jx_host);
    free(Jy_host);
    free(Jz_host);

    free(Mx_host);
    free(My_host);
    free(Mz_host);

    // Save the b data for future analysis
    std::ofstream outFile(std::string("data/b_N=") + std::to_string(N_TLS) + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS) 
        {
            Complex tmp_b = b_save_arr[id_TLS][iter];
            if (tmp_b.imag >= 0)
                outFile << tmp_b.real << "+" << tmp_b.imag << "j";
            else
                outFile << tmp_b.real << "-" << -tmp_b.imag << "j";
            if (id_TLS < N_TLS - 1)
                outFile << ", ";  // Separate using comma
        }
        outFile << std::endl;
    }
    outFile.close();

    // Save Ez sequence for future analysis
    outFile = std::ofstream(std::string("data/Ex") + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        float tmp_E_drive = Ex_save_arr[iter];
        outFile << tmp_E_drive << std::endl;
    }
    outFile.close();

    // Save the frequency monitor data for future analysis, if needed
    return;
}

void one_TLS_decay_PEC_w_box()
{
    // Constants
    const float um = 1.0f;

    // Wavelength basics
    float SOURCE_FREQUENCY = 1.0f;  // Frequency of the source
    float SOURCE_WAVELENGTH = (c0 / SOURCE_FREQUENCY);
    float SOURCE_OMEGA = 2 * PI * SOURCE_FREQUENCY;

    Complex eps_air = Complex(1.0f, 0.0f);

    // Parameters: TLS
    float d0 = 1e-2;  // Dipole moment
    float omega_TLS = 1.0 * SOURCE_OMEGA;  // TLS's resonance frequency
    // Frequency will shift due to self-interaction
    
    // Decay rate inside vacuum
    float Gamma0 = pow(d0, 2.0) * pow(omega_TLS, 3.0) / (3.0f * PI * hbar * eps0 * pow(c0, 3.0));
    printf("TLS decay rate: %.7f \n", Gamma0);

    // Grid resolution dx
    float dx = 0.025f;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    // float T = 1000 / SOURCE_FREQUENCY;  // 10 * T0
    int max_iter = 20000;  // Number of time-steps
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.2f;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);

    // Distance to PEC mirror
    float h_PEC = 0.1f;
    float h_PEC_grid = round(h_PEC / dx);
    h_PEC = h_PEC_grid * dx;
    printf("Distance to mirror: %.3f um \n", h_PEC);

    float Lx = 6 * um;
    float Ly = 6 * um;
    float Lz = 6 * um;
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

    // PEC: j > j_mid + h_PEC_grid
    int j_PEC = j_mid + h_PEC_grid;

    // Update the domain size
    Lx = (Nx - 1) * dx;
    Ly = (Ny - 1) * dx;
    Lz = (Nz - 1) * dx;

    // We need to define another domain for FDTD of radiation field
    int box_half = 1;  // Box size: 2 * box_half + 1
    int N_rad_half = box_half + t_PML + ceil(0.8 * SOURCE_WAVELENGTH / dx);  // + 1
    int N_rad = 2 * N_rad_half + 1;
    int N_rad_mid = N_rad_half + 1;  // The dipole position
    printf("Extra FDTD size: (%d, %d, %d) \n", N_rad, N_rad, N_rad);

    // Total size of 3D array
    size_t size = Nx * Ny * Nz * sizeof(float);
    size_t size_rad = N_rad * N_rad * N_rad * sizeof(float);

    // Now prepare for solving Maxwell's equations

    // Initialize arrays on device
    float *Ex, *Ey, *Ez;                      // On device
    float *Hx, *Hy, *Hz;                      // On device
    float *Jx, *Jy, *Jz;                      // On device
    float *Mx, *My, *Mz;                      // On device
    float *Cax, *Cbx, *Cay, *Cby, *Caz, *Cbz; // On device
    float *Dax, *Dbx, *Day, *Dby, *Daz, *Dbz; // On device

    // The radiation field
    float *Ex_rad, *Ey_rad, *Ez_rad;                      // On device
    float *Hx_rad, *Hy_rad, *Hz_rad;                      // On device
    float *Jx_rad, *Jy_rad, *Jz_rad;                      // On device
    float *Mx_rad, *My_rad, *Mz_rad;                      // On device
    float *Cax_rad, *Cbx_rad, *Cay_rad, *Cby_rad, *Caz_rad, *Cbz_rad; // On device
    float *Dax_rad, *Dbx_rad, *Day_rad, *Dby_rad, *Daz_rad, *Dbz_rad; // On device

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

    // Do similar things for radiation field
    // Allocate memory for the E and H fields on the device
    cuda_malloc_check((void **)&Ex_rad, size_rad);
    cuda_malloc_check((void **)&Ey_rad, size_rad);
    cuda_malloc_check((void **)&Ez_rad, size_rad);

    cuda_malloc_check((void **)&Hx_rad, size_rad);
    cuda_malloc_check((void **)&Hy_rad, size_rad);
    cuda_malloc_check((void **)&Hz_rad, size_rad);

    // Allocate memory for the J and M sources on the device
    cuda_malloc_check((void **)&Jx_rad, size_rad);
    cuda_malloc_check((void **)&Jy_rad, size_rad);
    cuda_malloc_check((void **)&Jz_rad, size_rad);

    cuda_malloc_check((void **)&Mx_rad, size_rad);
    cuda_malloc_check((void **)&My_rad, size_rad);
    cuda_malloc_check((void **)&Mz_rad, size_rad);

    // Allocate memory for the C and D matrices on the device
    cuda_malloc_check((void **)&Cax_rad, size_rad);
    cuda_malloc_check((void **)&Cbx_rad, size_rad);
    cuda_malloc_check((void **)&Cay_rad, size_rad);
    cuda_malloc_check((void **)&Cby_rad, size_rad);
    cuda_malloc_check((void **)&Caz_rad, size_rad);
    cuda_malloc_check((void **)&Cbz_rad, size_rad);

    cuda_malloc_check((void **)&Dax_rad, size_rad);
    cuda_malloc_check((void **)&Dbx_rad, size_rad);
    cuda_malloc_check((void **)&Day_rad, size_rad);
    cuda_malloc_check((void **)&Dby_rad, size_rad);
    cuda_malloc_check((void **)&Daz_rad, size_rad);
    cuda_malloc_check((void **)&Dbz_rad, size_rad);

    // Initialize fields & current sources with all zeros
    cudaMemset(Ex_rad, 0, size_rad);
    cudaMemset(Ey_rad, 0, size_rad);
    cudaMemset(Ez_rad, 0, size_rad);

    cudaMemset(Hx_rad, 0, size_rad);
    cudaMemset(Hy_rad, 0, size_rad);
    cudaMemset(Hz_rad, 0, size_rad);

    cudaMemset(Jx_rad, 0, size_rad);
    cudaMemset(Jy_rad, 0, size_rad);
    cudaMemset(Jz_rad, 0, size_rad);

    cudaMemset(Mx_rad, 0, size_rad);
    cudaMemset(My_rad, 0, size_rad);
    cudaMemset(Mz_rad, 0, size_rad);

    // Initialize arrays on host
    float *Jx_host, *Jy_host, *Jz_host;                      // On host
    float *Mx_host, *My_host, *Mz_host;                      // On host
    float *Cax_host, *Cbx_host, *Cay_host, *Cby_host, *Caz_host, *Cbz_host; // On host
    float *Dax_host, *Dbx_host, *Day_host, *Dby_host, *Daz_host, *Dbz_host; // On host

    // Allocate memory for the J and M sources on the host
    Jx_host = (float *)malloc(size);
    Jy_host = (float *)malloc(size);
    Jz_host = (float *)malloc(size);

    Mx_host = (float *)malloc(size);
    My_host = (float *)malloc(size);
    Mz_host = (float *)malloc(size);

    // Initialize current sources!
    memset(Jx_host, 0, size);
    memset(Jy_host, 0, size);
    memset(Jz_host, 0, size);
    
    memset(Mx_host, 0, size);
    memset(My_host, 0, size);
    memset(Mz_host, 0, size);

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
    
    // Also initialize structure for radiation field FDTD
    float *Cax_rad_host, *Cbx_rad_host, *Cay_rad_host, *Cby_rad_host, *Caz_rad_host, *Cbz_rad_host; // On host
    float *Dax_rad_host, *Dbx_rad_host, *Day_rad_host, *Dby_rad_host, *Daz_rad_host, *Dbz_rad_host; // On host

    // Allocate memory for the C matrices on the host
    Cax_rad_host = (float *)malloc(size_rad);
    Cbx_rad_host = (float *)malloc(size_rad);
    Cay_rad_host = (float *)malloc(size_rad);
    Cby_rad_host = (float *)malloc(size_rad);
    Caz_rad_host = (float *)malloc(size_rad);
    Cbz_rad_host = (float *)malloc(size_rad);

    // Allocate memory for the D matrices on the host
    Dax_rad_host = (float *)malloc(size_rad);
    Dbx_rad_host = (float *)malloc(size_rad);
    Day_rad_host = (float *)malloc(size_rad);
    Dby_rad_host = (float *)malloc(size_rad);
    Daz_rad_host = (float *)malloc(size_rad);
    Dbz_rad_host = (float *)malloc(size_rad);

    set_FDTD_matrices_3D(Cax_rad_host, Cbx_rad_host, Cay_rad_host, Cby_rad_host, 
        Caz_rad_host, Cbz_rad_host, Dax_rad_host, Dbx_rad_host, Day_rad_host, 
        Dby_rad_host, Daz_rad_host, Dbz_rad_host, 
        N_rad, N_rad, N_rad, dx, dt, eps_air, SOURCE_OMEGA, t_PML);

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

    cudaMemcpy(Cax_rad, Cax_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cbx_rad, Cbx_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cay_rad, Cay_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cby_rad, Cby_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Caz_rad, Caz_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cbz_rad, Cbz_rad_host, size_rad, cudaMemcpyHostToDevice);

    cudaMemcpy(Dax_rad, Dax_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dbx_rad, Dbx_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Day_rad, Day_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dby_rad, Dby_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Daz_rad, Daz_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dbz_rad, Dbz_rad_host, size_rad, cudaMemcpyHostToDevice);

    // Free these arrays on host, since we do not need them anymore.
    free(Cax_rad_host);
    free(Cbx_rad_host);
    free(Cay_rad_host);
    free(Cby_rad_host);
    free(Caz_rad_host);
    free(Cbz_rad_host);

    free(Dax_rad_host);
    free(Dbx_rad_host);
    free(Day_rad_host);
    free(Dby_rad_host);
    free(Daz_rad_host);
    free(Dbz_rad_host);

    // Allocate memory for field monitors
    float *E_monitor_xy, *E_rad_monitor_xy;
    E_monitor_xy = (float *)malloc(Nx * Ny * sizeof(float));
    E_rad_monitor_xy = (float *)malloc(N_rad * N_rad * sizeof(float));

    // TLS arrangement
    // In this case we put 1 TLS at the middle
    int N_TLS = 1;  // Total number of TLSs

    // Position of the TLS
    int i_TLS = i_mid;
    int j_TLS = j_mid;
    int k_TLS = k_mid;
    printf("TLS located at pixel (%d, %d, %d) \n", i_TLS, j_TLS, k_TLS);

    // Index of TLS
    int idx_TLS = i_TLS + j_TLS * Nx + k_TLS * (Nx * Ny);
    int idx_TLS_rad = N_rad_mid + N_rad_mid * N_rad + N_rad_mid * (N_rad * N_rad);

    // Indices for box boundaries [i1, i2] x [j1, j2] x [k1, k2]
    int i1 = i_mid - box_half; int i2 = i_mid + box_half;
    int j1 = j_mid - box_half; int j2 = j_mid + box_half;
    int k1 = k_mid - box_half; int k2 = k_mid + box_half;

    int i1_rad = N_rad_mid - box_half; int i2_rad = N_rad_mid + box_half;
    int j1_rad = N_rad_mid - box_half; int j2_rad = N_rad_mid + box_half;
    int k1_rad = N_rad_mid - box_half; int k2_rad = N_rad_mid + box_half;

    // Initialize TLS: start from excited state
    Complex b = Complex(1.0, 0.0);

    // We don't have extra current source now...
    
    // Initialize an array to save b value
    // Using vector of vectors for 2D array
    std::vector<std::vector<Complex>> b_save_arr(N_TLS, std::vector<Complex>(max_iter));
    
    std::vector<float> Ex_save_arr(max_iter);
    float tmp_E_drive = 0.0;  // To save Ez field at TLS position
    
    // Visualization settings
    float plot_field_bound = 0.05;  // Plot figure: colorbar bound
    int plot_interval = 500;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeRad((N_rad + blockSize.x - 1) / blockSize.x, (N_rad + blockSize.y - 1) / blockSize.y);

    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;

        // We will now carry out the 8 steps one-by-one!

        // Step 1: calculate J_{i}^{n} from b_{i}^{n}
        float Jx_value = 2 * d0 * omega_TLS * b.imag / pow(dx, 3);
        cudaMemcpy(Jx_rad + idx_TLS_rad, &Jx_value, sizeof(float), cudaMemcpyHostToDevice);

        // Step 2: update E_{i} based on J_{i}^{n}
        updateE<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, 
            Hx_rad, Hy_rad, Hz_rad, Cax_rad, Cbx_rad, Cay_rad, Cby_rad, 
            Caz_rad, Cbz_rad, Jx_rad, Jy_rad, Jz_rad, dx, N_rad, N_rad, N_rad);
        cudaDeviceSynchronize();

        // Step 3: calculate J_{i, rad}^{n} based on H_{i}^{n}
        // Equivalence principle for J
        // Set Jx, Jy, Jz
        
        // Face 1: i = i1
        // Jy[i1, j, k] += Hz_rad[i1-1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Jz[i1, j, k] += -Hy_rad[i1-1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Face 2: i = i2
        // Jy[i2, j, k] += -Hz_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Jz[i2, j, k] += Hy_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        set_J_rad_x_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 3: j = j1
        // Jx[i, j1, k] += -Hz_rad[i, j1-1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Jz[i, j1, k] += Hx_rad[i, j1-1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Face 4: j = j2
        // Jx[i, j2, k] += Hz_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Jz[i, j2, k] += -Hx_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        set_J_rad_y_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 5: k = k1
        // Jx[i, j, k1] += Hy_rad[i, j, k1-1] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Jy[i, j, k1] += -Hx_rad[i, j, k1-1] / dx for i in (i1, i2) and j in (j1, j2-1)
        // Face 6: k = k2
        // Jx[i, j, k2] += -Hy_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Jy[i, j, k2] += Hx_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
        set_J_rad_z_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Step 4: update E based on J_{i, rad}^{n}
        updateE_PEC<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, j_PEC, Nx, Ny, Nz);
        cudaDeviceSynchronize();
        // Don't forget to reset the current sources to all 0's!S
        cudaMemset(Jx, 0, size);
        cudaMemset(Jy, 0, size);
        cudaMemset(Jz, 0, size);

        // Step 5: calculate M_{i, rad}^{n+1/2} based on E_{i}^{n+1/2}
        // Equivalence principle for M
        // Set Mx, My and Mz
        
        // Face 1: i = i1
        // My[i1-1, j, k] += -Ez_rad[i1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Mz[i1-1, j, k] += Ey_rad[i1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Face 2: i = i2
        // My[i2, j, k] += Ez_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Mz[i2, j, k] += -Ey_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        set_M_rad_x_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Mx[i, j1-1, k] += Ez_rad[i, j1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Mz[i, j1-1, k] += -Ex_rad[i, j1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Face 4: j = j2
        // Mx[i, j2, k] += -Ez_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Mz[i, j2, k] += Ex_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        set_M_rad_y_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 5: k = k1
        // Mx[i, j, k1-1] += -Ey_rad[i, j, k1] / dx for i in (i1, i2) and j in (j1, j2-1)
        // My[i, j, k1-1] += Ex_rad[i, j, k1] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Face 6: k = k2
        // Mx[i, j, k2] += Ey_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
        // My[i, j, k2] += -Ex_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
        set_M_rad_z_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Step 6: update H based on M_{i, rad}^{n+1/2}
        updateH<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, 
            Mx, My, Mz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();
        // Don't forget to reset the current sources to all 0's!S
        cudaMemset(Mx, 0, size);
        cudaMemset(My, 0, size);
        cudaMemset(Mz, 0, size);

        // Step 7: update H_{i} based on H_{i}^{n}
        updateH<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, 
            Hx_rad, Hy_rad, Hz_rad, Dax_rad, Dbx_rad, Day_rad, 
            Dby_rad, Daz_rad, Dbz_rad, Mx_rad, My_rad, Mz_rad, 
            dx, N_rad, N_rad, N_rad);
        cudaDeviceSynchronize();

        // Step 8: update b
        // Get the field: copy Ez back to host, prepare to update bz
        cudaMemcpy(&tmp_E_drive, Ex + idx_TLS, sizeof(float), cudaMemcpyDeviceToHost);
        // b = Complex(0, 1) * Complex(cos(omega_TLS * t), -sin(omega_TLS * t)); // perfect oscillating dipole
        
        // Solve ODE and update b
        // Time interval: [t, t + dt]
        int ode_steps = 5;
        Complex b_new = b;
        for (int i_step = 0; i_step < ode_steps; ++i_step)
            b_new = RK4_step(t, b_new, dt / ode_steps, omega_TLS, d0, Gamma0, tmp_E_drive, true);
            
        // Record result
        b_save_arr[0][iter] = b_new;
        Ex_save_arr[iter] = tmp_E_drive;
        b = b_new;  // Update
        
        // Field monitor: update if needed

        // Frequency monitor: update the Fourier transform if needed
        
        // Record the E field using a monitor, once in a while
        if (iter % plot_interval == 0)
        {
            printf("Iter: %d / %d \n", iter, max_iter);
            
            // File name initialization
            char field_filename[50];
            
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
            
            snprintf(field_filename, sizeof(field_filename), "figures/Ex_%04d.png", iter);
            save_field_png(E_monitor_xy, field_filename, Nx, Ny, plot_field_bound);

            // Record: E_rad field at z=0 plane
            size_t slice_pitch_rad = N_rad * sizeof(float); // The size in bytes of the 2D slice row
            k = N_rad_mid;  // Assuming you want the middle slice

            for (int j = 0; j < N_rad; ++j)
            {
                float* device_ptr = Ex_rad + j * N_rad + k * N_rad * N_rad; // Pointer to the start of the row in the desired slice
                float* host_ptr = E_rad_monitor_xy + j * N_rad;  // Pointer to the host memory

                cudaMemcpy(host_ptr, device_ptr, slice_pitch_rad, cudaMemcpyDeviceToHost);
            }
            
            snprintf(field_filename, sizeof(field_filename), "figures/Ex_rad_%04d.png", iter);
            save_field_png(E_rad_monitor_xy, field_filename, N_rad, N_rad, plot_field_bound);
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

    // Don't forget radiation FDTD
    cudaFree(Ex_rad);
    cudaFree(Ey_rad);
    cudaFree(Ez_rad);

    cudaFree(Hx_rad);
    cudaFree(Hy_rad);
    cudaFree(Hz_rad);

    cudaFree(Jx_rad);
    cudaFree(Jy_rad);
    cudaFree(Jz_rad);

    cudaFree(Mx_rad);
    cudaFree(My_rad);
    cudaFree(Mz_rad);

    cudaFree(Cax_rad);
    cudaFree(Cbx_rad);
    cudaFree(Cay_rad);
    cudaFree(Cby_rad);
    cudaFree(Caz_rad);
    cudaFree(Cbz_rad);

    cudaFree(Dax_rad);
    cudaFree(Dbx_rad);
    cudaFree(Day_rad);
    cudaFree(Dby_rad);
    cudaFree(Daz_rad);
    cudaFree(Dbz_rad);

    // Free CPU memory
    free(Jx_host);
    free(Jy_host);
    free(Jz_host);

    free(Mx_host);
    free(My_host);
    free(Mz_host);

    // Save the b data for future analysis
    std::ofstream outFile(std::string("data/b_N=") + std::to_string(N_TLS) + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS) 
        {
            Complex tmp_b = b_save_arr[id_TLS][iter];
            if (tmp_b.imag >= 0)
                outFile << tmp_b.real << "+" << tmp_b.imag << "j";
            else
                outFile << tmp_b.real << "-" << -tmp_b.imag << "j";
            if (id_TLS < N_TLS - 1)
                outFile << ", ";  // Separate using comma
        }
        outFile << std::endl;
    }
    outFile.close();

    // Save Ez sequence for future analysis
    outFile = std::ofstream(std::string("data/Ex") + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        float tmp_E_drive = Ex_save_arr[iter];
        outFile << tmp_E_drive << std::endl;
    }
    outFile.close();

    // Save the frequency monitor data for future analysis, if needed
    return;
}

// Test Maxwell-Schrodinger equations, with PEC mirror
void Maxwell_Schrodinger_test()
{
    // Constants
    const float um = 1.0f;

    // Wavelength basics
    float SOURCE_FREQUENCY = 1.0f;  // Frequency of the source
    float SOURCE_WAVELENGTH = (c0 / SOURCE_FREQUENCY);
    float SOURCE_OMEGA = 2 * PI * SOURCE_FREQUENCY;

    Complex eps_air = Complex(1.0f, 0.0f);

    // Parameters: TLS
    float d0 = 1e-2;  // Dipole moment
    float omega_TLS = 1.0 * SOURCE_OMEGA;  // TLS's resonance frequency
    // Frequency will shift due to self-interaction
    
    // Decay rate inside vacuum
    float Gamma0 = pow(d0, 2.0) * pow(omega_TLS, 3.0) / (3.0f * PI * hbar * eps0 * pow(c0, 3.0));
    printf("TLS decay rate: %.7f \n", Gamma0);

    // Grid resolution dx
    float dx = 0.03f;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    float T = 300;  // 10 * T0
    int max_iter = ceil(T / dt);  // Number of time-steps
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.0f;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);

    // Distance to PEC mirror
    float h_PEC = 0.1f;
    float h_PEC_grid = round(h_PEC / dx);
    h_PEC = h_PEC_grid * dx;
    printf("Distance to mirror: %.3f um \n", h_PEC);

    float Lx = 6 * um;
    float Ly = 6 * um;
    float Lz = 6 * um;
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

    // PEC: j > j_mid + h_PEC_grid
    int j_PEC = j_mid + h_PEC_grid;

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

    // TLS arrangement
    // In this case we put 1 TLS at the middle
    int N_TLS = 1;  // Total number of TLSs in the array

    // Position of the TLS
    std::vector<int> idx_TLS_arr;
    idx_TLS_arr.push_back(i_mid + j_mid * Nx + k_mid * (Nx * Ny));

    std::vector<Complex> ce_arr;
    ce_arr.push_back(Complex(1.0, 0.0));  // Start from excited state
    std::vector<Complex> cg_arr;
    cg_arr.push_back(Complex(0.0 / sqrt(2.0), 0.0));

    // We don't have extra current source now...
    
    // Initialize an array to save Pe value
    // Using vector of vectors for 2D array
    std::vector<std::vector<float>> Pe_save_arr(N_TLS, std::vector<float>(max_iter));
    
    std::vector<float> Ex_save_arr(max_iter);
    float tmp_E_drive = 0.0;  // To save Ez field at TLS position
    
    // Visualization settings
    float plot_field_bound = 0.1;  // Plot figure: colorbar bound
    int plot_interval = 1000;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);

    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;
        
        // Update E fields
        updateE<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, Nx, Ny, Nz);
        /*
        updateE_PEC<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, j_PEC, Nx, Ny, Nz);
        */
        cudaDeviceSynchronize();

        // Update all TLSs
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            int idx_TLS = idx_TLS_arr[id_TLS];

            // Copy Ez back to host, prepare to update bz
            cudaMemcpy(&tmp_E_drive, Ex + idx_TLS, sizeof(float), cudaMemcpyDeviceToHost);

            Complex tmp_ce = ce_arr[id_TLS];
            Complex tmp_cg = cg_arr[id_TLS];

            // Solve ODE and update ce & cg
            // Time interval: [t, t + dt]
            int ode_steps = 5;
            Complex ce_new = tmp_ce; Complex cg_new = tmp_cg;

            for (int i_step = 0; i_step < ode_steps; ++i_step)
            {
                /*
                // Wrong method: Euler method
                Complex dce_dt = Complex(0.0, -1.0) * omega_TLS * ce_new + 
                    Complex(0.0, 1.0) * (d0 * tmp_E_drive / hbar) * cg_new;
                Complex dcg_dt = Complex(0.0, 1.0) * (d0 * tmp_E_drive / hbar) * ce_new;

                // Euler method update
                ce_new = tmp_ce + dce_dt * dt / ode_steps;
                cg_new = tmp_cg + dcg_dt * dt / ode_steps;
                */
                // Correct: RK4
                std::pair<Complex, Complex> results = RK4_Schrodinger_step(ce_new, cg_new, 
                    omega_TLS, d0, tmp_E_drive, dt / ode_steps);
                ce_new = results.first;
                cg_new = results.second;
            }
            
            // Record result
            Pe_save_arr[id_TLS][iter] = pow(ce_new.abs(), 2);
            Ex_save_arr[iter] = tmp_E_drive;
            
            // Update the J current sources
            // The original version
            Complex Jx_value_complex = (ce_new * cg_new.conj() + ce_new.conj() * cg_new) - 
                (tmp_ce * tmp_cg.conj() + tmp_ce.conj() * tmp_cg);
            float Jx_value = Jx_value_complex.real;
            Jx_value = -Jx_value * d0 / dt / pow(dx, 3);
            // It indeed should be dx^3!

            // Copy the J current source to device
            cudaMemcpy(Jx + idx_TLS, &Jx_value, sizeof(float), cudaMemcpyHostToDevice);
            
            // Update TLS status by replacing ce, cg
            ce_arr[id_TLS] = ce_new;  // ce_new
            cg_arr[id_TLS] = cg_new;  // cg_new
        }
        
        // Update H fields
        updateH<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, 
            Mx, My, Mz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();

        // Field monitor: update if needed

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

            snprintf(field_filename, sizeof(field_filename), "figures/Ex_Schrodinger_%04d.png", iter);
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
    std::ofstream outFile(std::string("data/Pe_Schrodinger.csv"));
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
    outFile = std::ofstream(std::string("data/Ex") + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        float tmp_E_drive = Ex_save_arr[iter];
        outFile << tmp_E_drive << std::endl;
    }
    outFile.close();

    // Save the frequency monitor data for future analysis, if needed
    return;
}

// Test Maxwell-Bloch equations, with PEC mirror
void Maxwell_Bloch_test()
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
    float dx = 0.03f;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    float T = 300;  // 10 * T0
    int max_iter = ceil(T / dt);  // Number of time-steps
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.0f;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);

    // Distance to PEC mirror
    float h_PEC = 0.1f;
    float h_PEC_grid = round(h_PEC / dx);
    h_PEC = h_PEC_grid * dx;
    printf("Distance to mirror: %.3f um \n", h_PEC);

    float Lx = 6 * um;
    float Ly = 6 * um;
    float Lz = 6 * um;
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

    // PEC: j > j_mid + h_PEC_grid
    int j_PEC = j_mid + h_PEC_grid;

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

    // TLS arrangement
    // In this case we put 1 TLS at the middle
    int N_TLS = 1;  // Total number of TLSs in the array

    // Position of the TLS
    std::vector<int> idx_TLS_arr;
    idx_TLS_arr.push_back(i_mid + j_mid * Nx + k_mid * (Nx * Ny));

    std::vector<Complex> rho_ee_arr;
    rho_ee_arr.push_back(Complex(1.0, 0.0));  // Start from excited state
    std::vector<Complex> rho_eg_arr;
    rho_eg_arr.push_back(Complex(0.0, 0.0));

    // We don't have extra current source now...
    
    // Initialize an array to save Pe value
    // Using vector of vectors for 2D array
    std::vector<std::vector<float>> Pe_save_arr(N_TLS, std::vector<float>(max_iter));
    
    std::vector<float> Ex_save_arr(max_iter);
    float tmp_E_drive = 0.0;  // To save Ez field at TLS position
    
    // Visualization settings
    float plot_field_bound = 0.1;  // Plot figure: colorbar bound
    int plot_interval = 1000;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);

    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;
        
        // Update E fields
        updateE<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, Nx, Ny, Nz);
        /*
        updateE_PEC<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, j_PEC, Nx, Ny, Nz);
        */
        cudaDeviceSynchronize();

        // Update all TLSs
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            int idx_TLS = idx_TLS_arr[id_TLS];

            // Copy Ez back to host, prepare to update bz
            cudaMemcpy(&tmp_E_drive, Ex + idx_TLS, sizeof(float), cudaMemcpyDeviceToHost);

            Complex tmp_rho_ee = rho_ee_arr[id_TLS];
            Complex tmp_rho_eg = rho_eg_arr[id_TLS];

            // Solve ODE and update ce & cg
            // Time interval: [t, t + dt]
            int ode_steps = 5;
            Complex rho_ee_new = tmp_rho_ee; Complex rho_eg_new = tmp_rho_eg;

            for (int i_step = 0; i_step < ode_steps; ++i_step)
            {
                // Correct: RK4
                std::pair<Complex, Complex> results = RK4_Bloch_step(rho_ee_new, rho_eg_new, 
                    omega_TLS, d0, tmp_E_drive, Gamma0, dt / ode_steps);
                rho_ee_new = results.first;
                rho_eg_new = results.second;
            }
            
            // Record result
            Pe_save_arr[id_TLS][iter] = rho_ee_new.abs();
            Ex_save_arr[iter] = tmp_E_drive;
            
            // Update the J current sources
            // The original version
            Complex Jx_value_complex = (rho_eg_new + rho_eg_new.conj()) - 
                (tmp_rho_eg + tmp_rho_eg.conj());
            float Jx_value = Jx_value_complex.real;
            Jx_value = Jx_value * d0 / dt / pow(dx, 3);
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

        // Field monitor: update if needed

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

            snprintf(field_filename, sizeof(field_filename), "figures/Ex_Bloch_%04d.png", iter);
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
    std::ofstream outFile(std::string("data/Pe_Bloch.csv"));
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
    outFile = std::ofstream(std::string("data/Ex") + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        float tmp_E_drive = Ex_save_arr[iter];
        outFile << tmp_E_drive << std::endl;
    }
    outFile.close();

    // Save the frequency monitor data for future analysis, if needed
    return;
}

// Produce a figure to explain how our technique works
void one_TLS_pulse_PEC_w_box()
{
    // Constants
    const float um = 1.0f;

    // Wavelength basics
    float SOURCE_FREQUENCY = 1.0f;  // Frequency of the source
    float SOURCE_WAVELENGTH = (c0 / SOURCE_FREQUENCY);
    float SOURCE_OMEGA = 2 * PI * SOURCE_FREQUENCY;

    Complex eps_air = Complex(1.0f, 0.0f);

    // Parameters: TLS
    float d0 = 1e-2;  // Dipole moment
    float omega_TLS = 1.0 * SOURCE_OMEGA;  // TLS's resonance frequency
    float T0 = 2 * PI / omega_TLS;
    // Frequency will shift due to self-interaction
    
    // Decay rate inside vacuum
    float Gamma0 = pow(d0, 2.0) * pow(omega_TLS, 3.0) / (3.0f * PI * hbar * eps0 * pow(c0, 3.0));
    printf("TLS decay rate: %.7f \n", Gamma0);

    // Grid resolution dx
    float dx = 0.04f;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    int max_iter = 700;  // Number of time-steps
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.2f;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);

    // Distance to PEC mirror
    float h_PEC = 3.5f;
    float h_PEC_grid = round(h_PEC / dx);
    h_PEC = h_PEC_grid * dx;
    printf("Distance to mirror: %.3f um \n", h_PEC);
    
    float Lx = 10 * um;
    float Ly = 10 * um;
    float Lz = 6 * um;
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

    // PEC: j > j_mid + h_PEC_grid
    int j_PEC = j_mid + h_PEC_grid;

    // Update the domain size
    Lx = (Nx - 1) * dx;
    Ly = (Ny - 1) * dx;
    Lz = (Nz - 1) * dx;

    // We need to define another domain for FDTD of radiation field
    int box_half = 20;  // Box size: 2 * box_half + 1
    int N_rad_half = box_half + t_PML + ceil(0 * SOURCE_WAVELENGTH / dx);  // + 1
    int N_rad = 2 * N_rad_half + 1;
    int N_rad_mid = N_rad_half + 1;  // The dipole position
    printf("Extra FDTD size: (%d, %d, %d) \n", N_rad, N_rad, N_rad);

    // Total size of 3D array
    size_t size = Nx * Ny * Nz * sizeof(float);
    size_t size_rad = N_rad * N_rad * N_rad * sizeof(float);

    // Now prepare for solving Maxwell's equations

    // Initialize arrays on device
    float *Ex, *Ey, *Ez;                      // On device
    float *Hx, *Hy, *Hz;                      // On device
    float *Jx, *Jy, *Jz;                      // On device
    float *Mx, *My, *Mz;                      // On device
    float *Cax, *Cbx, *Cay, *Cby, *Caz, *Cbz; // On device
    float *Dax, *Dbx, *Day, *Dby, *Daz, *Dbz; // On device

    // The radiation field
    float *Ex_rad, *Ey_rad, *Ez_rad;                      // On device
    float *Hx_rad, *Hy_rad, *Hz_rad;                      // On device
    float *Jx_rad, *Jy_rad, *Jz_rad;                      // On device
    float *Mx_rad, *My_rad, *Mz_rad;                      // On device
    float *Cax_rad, *Cbx_rad, *Cay_rad, *Cby_rad, *Caz_rad, *Cbz_rad; // On device
    float *Dax_rad, *Dbx_rad, *Day_rad, *Dby_rad, *Daz_rad, *Dbz_rad; // On device

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

    // Do similar things for radiation field
    // Allocate memory for the E and H fields on the device
    cuda_malloc_check((void **)&Ex_rad, size_rad);
    cuda_malloc_check((void **)&Ey_rad, size_rad);
    cuda_malloc_check((void **)&Ez_rad, size_rad);

    cuda_malloc_check((void **)&Hx_rad, size_rad);
    cuda_malloc_check((void **)&Hy_rad, size_rad);
    cuda_malloc_check((void **)&Hz_rad, size_rad);

    // Allocate memory for the J and M sources on the device
    cuda_malloc_check((void **)&Jx_rad, size_rad);
    cuda_malloc_check((void **)&Jy_rad, size_rad);
    cuda_malloc_check((void **)&Jz_rad, size_rad);

    cuda_malloc_check((void **)&Mx_rad, size_rad);
    cuda_malloc_check((void **)&My_rad, size_rad);
    cuda_malloc_check((void **)&Mz_rad, size_rad);

    // Allocate memory for the C and D matrices on the device
    cuda_malloc_check((void **)&Cax_rad, size_rad);
    cuda_malloc_check((void **)&Cbx_rad, size_rad);
    cuda_malloc_check((void **)&Cay_rad, size_rad);
    cuda_malloc_check((void **)&Cby_rad, size_rad);
    cuda_malloc_check((void **)&Caz_rad, size_rad);
    cuda_malloc_check((void **)&Cbz_rad, size_rad);

    cuda_malloc_check((void **)&Dax_rad, size_rad);
    cuda_malloc_check((void **)&Dbx_rad, size_rad);
    cuda_malloc_check((void **)&Day_rad, size_rad);
    cuda_malloc_check((void **)&Dby_rad, size_rad);
    cuda_malloc_check((void **)&Daz_rad, size_rad);
    cuda_malloc_check((void **)&Dbz_rad, size_rad);

    // Initialize fields & current sources with all zeros
    cudaMemset(Ex_rad, 0, size_rad);
    cudaMemset(Ey_rad, 0, size_rad);
    cudaMemset(Ez_rad, 0, size_rad);

    cudaMemset(Hx_rad, 0, size_rad);
    cudaMemset(Hy_rad, 0, size_rad);
    cudaMemset(Hz_rad, 0, size_rad);

    cudaMemset(Jx_rad, 0, size_rad);
    cudaMemset(Jy_rad, 0, size_rad);
    cudaMemset(Jz_rad, 0, size_rad);

    cudaMemset(Mx_rad, 0, size_rad);
    cudaMemset(My_rad, 0, size_rad);
    cudaMemset(Mz_rad, 0, size_rad);

    // Initialize arrays on host
    float *Jx_host, *Jy_host, *Jz_host;                      // On host
    float *Mx_host, *My_host, *Mz_host;                      // On host
    float *Cax_host, *Cbx_host, *Cay_host, *Cby_host, *Caz_host, *Cbz_host; // On host
    float *Dax_host, *Dbx_host, *Day_host, *Dby_host, *Daz_host, *Dbz_host; // On host

    // Allocate memory for the J and M sources on the host
    Jx_host = (float *)malloc(size);
    Jy_host = (float *)malloc(size);
    Jz_host = (float *)malloc(size);

    Mx_host = (float *)malloc(size);
    My_host = (float *)malloc(size);
    Mz_host = (float *)malloc(size);

    // Initialize current sources!
    memset(Jx_host, 0, size);
    memset(Jy_host, 0, size);
    memset(Jz_host, 0, size);
    
    memset(Mx_host, 0, size);
    memset(My_host, 0, size);
    memset(Mz_host, 0, size);

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
    
    // Also initialize structure for radiation field FDTD
    float *Cax_rad_host, *Cbx_rad_host, *Cay_rad_host, *Cby_rad_host, *Caz_rad_host, *Cbz_rad_host; // On host
    float *Dax_rad_host, *Dbx_rad_host, *Day_rad_host, *Dby_rad_host, *Daz_rad_host, *Dbz_rad_host; // On host

    // Allocate memory for the C matrices on the host
    Cax_rad_host = (float *)malloc(size_rad);
    Cbx_rad_host = (float *)malloc(size_rad);
    Cay_rad_host = (float *)malloc(size_rad);
    Cby_rad_host = (float *)malloc(size_rad);
    Caz_rad_host = (float *)malloc(size_rad);
    Cbz_rad_host = (float *)malloc(size_rad);

    // Allocate memory for the D matrices on the host
    Dax_rad_host = (float *)malloc(size_rad);
    Dbx_rad_host = (float *)malloc(size_rad);
    Day_rad_host = (float *)malloc(size_rad);
    Dby_rad_host = (float *)malloc(size_rad);
    Daz_rad_host = (float *)malloc(size_rad);
    Dbz_rad_host = (float *)malloc(size_rad);

    set_FDTD_matrices_3D(Cax_rad_host, Cbx_rad_host, Cay_rad_host, Cby_rad_host, 
        Caz_rad_host, Cbz_rad_host, Dax_rad_host, Dbx_rad_host, Day_rad_host, 
        Dby_rad_host, Daz_rad_host, Dbz_rad_host, 
        N_rad, N_rad, N_rad, dx, dt, eps_air, SOURCE_OMEGA, t_PML);

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

    cudaMemcpy(Cax_rad, Cax_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cbx_rad, Cbx_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cay_rad, Cay_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cby_rad, Cby_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Caz_rad, Caz_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Cbz_rad, Cbz_rad_host, size_rad, cudaMemcpyHostToDevice);

    cudaMemcpy(Dax_rad, Dax_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dbx_rad, Dbx_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Day_rad, Day_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dby_rad, Dby_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Daz_rad, Daz_rad_host, size_rad, cudaMemcpyHostToDevice);
    cudaMemcpy(Dbz_rad, Dbz_rad_host, size_rad, cudaMemcpyHostToDevice);

    // Free these arrays on host, since we do not need them anymore.
    free(Cax_rad_host);
    free(Cbx_rad_host);
    free(Cay_rad_host);
    free(Cby_rad_host);
    free(Caz_rad_host);
    free(Cbz_rad_host);

    free(Dax_rad_host);
    free(Dbx_rad_host);
    free(Day_rad_host);
    free(Dby_rad_host);
    free(Daz_rad_host);
    free(Dbz_rad_host);

    // Allocate memory for field monitors
    float *E_monitor_xy, *E_rad_monitor_xy;
    E_monitor_xy = (float *)malloc(Nx * Ny * sizeof(float));
    E_rad_monitor_xy = (float *)malloc(N_rad * N_rad * sizeof(float));

    // TLS arrangement
    // In this case we put 1 TLS at the middle
    int N_TLS = 1;  // Total number of TLSs

    // Position of the TLS
    int i_TLS = i_mid;
    int j_TLS = j_mid;
    int k_TLS = k_mid;
    printf("TLS located at pixel (%d, %d, %d) \n", i_TLS, j_TLS, k_TLS);

    // Index of TLS
    int idx_TLS = i_TLS + j_TLS * Nx + k_TLS * (Nx * Ny);
    int idx_TLS_rad = N_rad_mid + N_rad_mid * N_rad + N_rad_mid * (N_rad * N_rad);

    // Indices for box boundaries [i1, i2] x [j1, j2] x [k1, k2]
    int i1 = i_mid - box_half; int i2 = i_mid + box_half;
    int j1 = j_mid - box_half; int j2 = j_mid + box_half;
    int k1 = k_mid - box_half; int k2 = k_mid + box_half;

    int i1_rad = N_rad_mid - box_half; int i2_rad = N_rad_mid + box_half;
    int j1_rad = N_rad_mid - box_half; int j2_rad = N_rad_mid + box_half;
    int k1_rad = N_rad_mid - box_half; int k2_rad = N_rad_mid + box_half;

    // Initialize TLS: start from excited state
    Complex b = Complex(1.0, 0.0);

    // We don't have extra current source now...
    
    // Initialize an array to save b value
    // Using vector of vectors for 2D array
    std::vector<std::vector<Complex>> b_save_arr(N_TLS, std::vector<Complex>(max_iter));
    
    std::vector<float> Ex_save_arr(max_iter);
    float tmp_E_drive = 0.0;  // To save Ez field at TLS position
    
    // Visualization settings
    float plot_field_bound = 0.01;  // Plot figure: colorbar bound
    int plot_interval = 10;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeRad((N_rad + blockSize.x - 1) / blockSize.x, (N_rad + blockSize.y - 1) / blockSize.y);

    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;

        // We will now carry out the 8 steps one-by-one!

        // Step 1: calculate J_{i}^{n} from b_{i}^{n}
        float Jx_value = 2 * d0 * omega_TLS * b.imag / pow(dx, 3);
        cudaMemcpy(Jx_rad + idx_TLS_rad, &Jx_value, sizeof(float), cudaMemcpyHostToDevice);

        // Step 2: update E_{i} based on J_{i}^{n}
        updateE<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, 
            Hx_rad, Hy_rad, Hz_rad, Cax_rad, Cbx_rad, Cay_rad, Cby_rad, 
            Caz_rad, Cbz_rad, Jx_rad, Jy_rad, Jz_rad, dx, N_rad, N_rad, N_rad);
        cudaDeviceSynchronize();

        // Step 3: calculate J_{i, rad}^{n} based on H_{i}^{n}
        // Equivalence principle for J
        // Set Jx, Jy, Jz
        
        // Face 1: i = i1
        // Jy[i1, j, k] += Hz_rad[i1-1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Jz[i1, j, k] += -Hy_rad[i1-1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Face 2: i = i2
        // Jy[i2, j, k] += -Hz_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Jz[i2, j, k] += Hy_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        set_J_rad_x_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 3: j = j1
        // Jx[i, j1, k] += -Hz_rad[i, j1-1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Jz[i, j1, k] += Hx_rad[i, j1-1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Face 4: j = j2
        // Jx[i, j2, k] += Hz_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Jz[i, j2, k] += -Hx_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        set_J_rad_y_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 5: k = k1
        // Jx[i, j, k1] += Hy_rad[i, j, k1-1] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Jy[i, j, k1] += -Hx_rad[i, j, k1-1] / dx for i in (i1, i2) and j in (j1, j2-1)
        // Face 6: k = k2
        // Jx[i, j, k2] += -Hy_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Jy[i, j, k2] += Hx_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
        set_J_rad_z_dir<<<gridSizeRad, blockSize>>>(Hx_rad, Hy_rad, Hz_rad, Jx, Jy, Jz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Step 4: update E based on J_{i, rad}^{n}
        updateE_PEC<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, j_PEC, Nx, Ny, Nz);
        cudaDeviceSynchronize();
        // Don't forget to reset the current sources to all 0's!S
        cudaMemset(Jx, 0, size);
        cudaMemset(Jy, 0, size);
        cudaMemset(Jz, 0, size);

        // Step 5: calculate M_{i, rad}^{n+1/2} based on E_{i}^{n+1/2}
        // Equivalence principle for M
        // Set Mx, My and Mz
        
        // Face 1: i = i1
        // My[i1-1, j, k] += -Ez_rad[i1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Mz[i1-1, j, k] += Ey_rad[i1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        // Face 2: i = i2
        // My[i2, j, k] += Ez_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
        // Mz[i2, j, k] += -Ey_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
        set_M_rad_x_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Mx[i, j1-1, k] += Ez_rad[i, j1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Mz[i, j1-1, k] += -Ex_rad[i, j1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        // Face 4: j = j2
        // Mx[i, j2, k] += -Ez_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
        // Mz[i, j2, k] += Ex_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
        set_M_rad_y_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Face 5: k = k1
        // Mx[i, j, k1-1] += -Ey_rad[i, j, k1] / dx for i in (i1, i2) and j in (j1, j2-1)
        // My[i, j, k1-1] += Ex_rad[i, j, k1] / dx for i in (i1, i2-1) and j in (j1, j2)
        // Face 6: k = k2
        // Mx[i, j, k2] += Ey_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
        // My[i, j, k2] += -Ex_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
        set_M_rad_z_dir<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, Mx, My, Mz, 
            dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
            j1_rad, j2_rad, k1_rad, k2_rad);
        cudaDeviceSynchronize();
        
        // Step 6: update H based on M_{i, rad}^{n+1/2}
        updateH<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, 
            Mx, My, Mz, dx, Nx, Ny, Nz);
        cudaDeviceSynchronize();
        // Don't forget to reset the current sources to all 0's!S
        cudaMemset(Mx, 0, size);
        cudaMemset(My, 0, size);
        cudaMemset(Mz, 0, size);

        // Step 7: update H_{i} based on H_{i}^{n}
        updateH<<<gridSizeRad, blockSize>>>(Ex_rad, Ey_rad, Ez_rad, 
            Hx_rad, Hy_rad, Hz_rad, Dax_rad, Dbx_rad, Day_rad, 
            Dby_rad, Daz_rad, Dbz_rad, Mx_rad, My_rad, Mz_rad, 
            dx, N_rad, N_rad, N_rad);
        cudaDeviceSynchronize();

        // Step 8: update b
        // Get the field: copy Ez back to host, prepare to update bz
        cudaMemcpy(&tmp_E_drive, Ex + idx_TLS, sizeof(float), cudaMemcpyDeviceToHost);
        
        b = Complex(0, 0.5) * exp(-pow(t - 3 * T0, 2) / (2 * T0 * T0));
        b = b * Complex(cos(omega_TLS * t), -sin(omega_TLS * t)); // use a pulse
        
        // Field monitor: update if needed

        // Frequency monitor: update the Fourier transform if needed
        
        // Record the E field using a monitor, once in a while
        if (iter % plot_interval == 0)
        {
            printf("Iter: %d / %d \n", iter, max_iter);
            
            // File name initialization
            char field_filename[50];
            
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
            
            snprintf(field_filename, sizeof(field_filename), "figures/Ex_%04d.png", iter);
            save_field_png(E_monitor_xy, field_filename, Nx, Ny, plot_field_bound);

            // Record: E_rad field at z=0 plane
            size_t slice_pitch_rad = N_rad * sizeof(float); // The size in bytes of the 2D slice row
            k = N_rad_mid;  // Assuming you want the middle slice

            for (int j = 0; j < N_rad; ++j)
            {
                float* device_ptr = Ex_rad + j * N_rad + k * N_rad * N_rad; // Pointer to the start of the row in the desired slice
                float* host_ptr = E_rad_monitor_xy + j * N_rad;  // Pointer to the host memory

                cudaMemcpy(host_ptr, device_ptr, slice_pitch_rad, cudaMemcpyDeviceToHost);
            }
            
            snprintf(field_filename, sizeof(field_filename), "figures/Ex_rad_%04d.png", iter);
            save_field_png(E_rad_monitor_xy, field_filename, N_rad, N_rad, plot_field_bound);
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

    // Don't forget radiation FDTD
    cudaFree(Ex_rad);
    cudaFree(Ey_rad);
    cudaFree(Ez_rad);

    cudaFree(Hx_rad);
    cudaFree(Hy_rad);
    cudaFree(Hz_rad);

    cudaFree(Jx_rad);
    cudaFree(Jy_rad);
    cudaFree(Jz_rad);

    cudaFree(Mx_rad);
    cudaFree(My_rad);
    cudaFree(Mz_rad);

    cudaFree(Cax_rad);
    cudaFree(Cbx_rad);
    cudaFree(Cay_rad);
    cudaFree(Cby_rad);
    cudaFree(Caz_rad);
    cudaFree(Cbz_rad);

    cudaFree(Dax_rad);
    cudaFree(Dbx_rad);
    cudaFree(Day_rad);
    cudaFree(Dby_rad);
    cudaFree(Daz_rad);
    cudaFree(Dbz_rad);

    // Free CPU memory
    free(Jx_host);
    free(Jy_host);
    free(Jz_host);

    free(Mx_host);
    free(My_host);
    free(Mz_host);

    // Save the b data for future analysis
    std::ofstream outFile(std::string("data/b_N=") + std::to_string(N_TLS) + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS) 
        {
            Complex tmp_b = b_save_arr[id_TLS][iter];
            if (tmp_b.imag >= 0)
                outFile << tmp_b.real << "+" << tmp_b.imag << "j";
            else
                outFile << tmp_b.real << "-" << -tmp_b.imag << "j";
            if (id_TLS < N_TLS - 1)
                outFile << ", ";  // Separate using comma
        }
        outFile << std::endl;
    }
    outFile.close();

    // Save Ez sequence for future analysis
    outFile = std::ofstream(std::string("data/Ex") + std::string(".csv"));
    for (int iter = 0; iter < max_iter; ++iter) 
    {
        float tmp_E_drive = Ex_save_arr[iter];
        outFile << tmp_E_drive << std::endl;
    }
    outFile.close();

    // Save the frequency monitor data for future analysis, if needed
    return;
}

int main()
{
    // Spontaneous emission: 1 TLS inside vacuum
    // one_TLS_decay_vacuum_wo_box();  // No box

    // Spontaneous emission: 1 TLS inside vacuum
    // one_TLS_decay_vacuum_w_box();  // Enclosing TLS with box

    // Spontaneous emission: 1 TLS with PEC mirror
    // one_TLS_decay_PEC_wo_box();  // No box

    // Spontaneous emission: 1 TLS with PEC mirror
    // one_TLS_decay_PEC_w_box();  // Enclosing TLS with box

    // Maxwell-Schrodinger equation: 1 TLS with PEC
    // Maxwell_Schrodinger_test();

    // Maxwell-Bloch equation: 1 TLS with PEC
    // Maxwell_Bloch_test();

    // Produce a figure to illustrate TF-IF
    one_TLS_pulse_PEC_w_box();

    return 0;
}
