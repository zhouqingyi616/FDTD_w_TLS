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

// FDTD update rule for E field, with mode injection
__global__ void updateE_mode(float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, 
    float *Cax, float *Cbx, float *Cay, float *Cby, float *Caz, float *Cbz, 
    float *Jx, float *Jy, float *Jz, float dx, int Nx, int Ny, int Nz, 
    int i_inc, int j_inc_start, int j_inc_end, int k_inc_start, int k_inc_end, 
    float *ReHy_mode, float *ImHy_mode, float *ReHz_mode, float *ImHz_mode, 
    int Nz_wg_mode, float amp, float omega, float t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1) 
    {
        // Iterate over z direction
        for (int k = 1; k < Nz - 1; ++k) 
        {
            int idx = i + j * Nx + k * (Nx * Ny);

            // Use mode source to modify J source
            if (i == i_inc && j >= j_inc_start && j <= j_inc_end && k >= k_inc_start && k <= k_inc_end)
            {
                int j_mode_source = j - j_inc_start;
                int k_mode_source = k - k_inc_start;

                int idx_mode_source = j_mode_source * Nz_wg_mode + k_mode_source;

                // Jy ~ -Hz / dx, Jz ~ Hy / dx
                Jy[idx] = -(ReHz_mode[idx_mode_source] * cos(omega * t) + 
                    ImHz_mode[idx_mode_source] * sin(omega * t)) * amp / dx;
                    
                Jz[idx] = (ReHy_mode[idx_mode_source] * cos(omega * t) + 
                    ImHy_mode[idx_mode_source] * sin(omega * t)) * amp / dx;
                    
            }

            Ex[idx] = Cax[idx] * Ex[idx] + Cbx[idx] * 
                      ((Hz[idx] - Hz[idx - Nx]) / dx - (Hy[idx] - Hy[idx - Nx * Ny]) / dx - Jx[idx]);
                      
            Ey[idx] = Cay[idx] * Ey[idx] + Cby[idx] * 
                      ((Hx[idx] - Hx[idx - Nx * Ny]) / dx - (Hz[idx] - Hz[idx - 1]) / dx - Jy[idx]);

            Ez[idx] = Caz[idx] * Ez[idx] + Cbz[idx] * 
                      ((Hy[idx] - Hy[idx - 1]) / dx - (Hx[idx] - Hx[idx - Nx]) / dx - Jz[idx]);
        }
    }
}

// FDTD update rule for H field, with mode injection
__global__ void updateH_mode(float *Ex, float *Ey, float *Ez, float *Hx, float *Hy, float *Hz, 
    float *Dax, float *Dbx, float *Day, float *Dby, float *Daz, float *Dbz, 
    float *Mx, float *My, float *Mz, float dx, int Nx, int Ny, int Nz, 
    int i_inc, int j_inc_start, int j_inc_end, int k_inc_start, int k_inc_end, 
    float *ReEy_mode, float *ImEy_mode, float *ReEz_mode, float *ImEz_mode, 
    int Nz_wg_mode, float amp, float omega, float t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1) 
    {
        // Iterate over z direction
        for (int k = 1; k < Nz - 1; ++k)
        {
            int idx = i + j * Nx + k * (Nx * Ny);

            // Use mode source to modify M source
            if (i == i_inc && j >= j_inc_start && j <= j_inc_end && k >= k_inc_start && k <= k_inc_end)
            {
                int j_mode_source = j - j_inc_start;
                int k_mode_source = k - k_inc_start;

                int idx_mode_source = j_mode_source * Nz_wg_mode + k_mode_source;
                
                // My ~ Ez / dx, Mz ~ -Ey / dx
                My[idx] = (ReEz_mode[idx_mode_source] * cos(omega * t) + 
                    ImEz_mode[idx_mode_source] * sin(omega * t)) * amp / dx;
                
                Mz[idx] = -(ReEy_mode[idx_mode_source] * cos(omega * t) + 
                    ImEy_mode[idx_mode_source] * sin(omega * t)) * amp / dx;
            }
            
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
    float *Jx, float *Jy, float *Jz, float dx, int Nx, int Ny, int Nz, int N_rad, 
    int i1, int i2, int j1, int j2, int k1, int k2, int i1_rad, int i2_rad, 
    int j1_rad, int j2_rad, int k1_rad, int k2_rad)
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
    float *Jx, float *Jy, float *Jz, float dx, int Nx, int Ny, int Nz, int N_rad, 
    int i1, int i2, int j1, int j2, int k1, int k2, int i1_rad, int i2_rad, 
    int j1_rad, int j2_rad, int k1_rad, int k2_rad)
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
    float *Jx, float *Jy, float *Jz, float dx, int Nx, int Ny, int Nz, int N_rad, 
    int i1, int i2, int j1, int j2, int k1, int k2, int i1_rad, int i2_rad, 
    int j1_rad, int j2_rad, int k1_rad, int k2_rad)
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

// Utility function for copying vector<vector<float>> to GPU
float* copyToCudaDevice(const std::vector<std::vector<float>>& data, int rows, int cols) 
{
    // Step 1: Flatten the 2D vector into a contiguous 1D array
    std::vector<float> flatData;
    flatData.reserve(rows * cols);
    for (const auto& row : data)
        flatData.insert(flatData.end(), row.begin(), row.end());

    // Step 2: Allocate memory on the CUDA device
    float* d_data;
    cudaMalloc(&d_data, flatData.size() * sizeof(float));

    // Step 3: Copy data from the host to the CUDA device
    cudaMemcpy(d_data, flatData.data(), flatData.size() * sizeof(float), cudaMemcpyHostToDevice);

    return d_data;  // Return the device pointer
}

// Utility function for collecting monitor data
__global__ void extract_field_slice(float *d_E, float *d_monitor, int i0, int j0, int j1, 
    int k0, int k1, int Nx, int Ny) 
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + j0;
    int k = blockIdx.y * blockDim.y + threadIdx.y + k0;

    if (j <= j1 && k <= k1) 
    {
        int idx_in_full = i0 + j * Nx + k * Nx * Ny;  // index in the full array
        int idx_in_sub = (j - j0) * (k1 - k0 + 1) + (k - k0); // index in the subregion array
        d_monitor[idx_in_sub] = d_E[idx_in_full];
    }
}

// Utility function for allocating memory & initialization of 6 arrays
inline void cuda_malloc_memset_6_arrays(float** A, float** B, float** C, 
    float** D, float** E, float** F, size_t size)
{
    cuda_malloc_check((void **)A, size);
    cudaMemset(*A, 0, size);

    cuda_malloc_check((void **)B, size);
    cudaMemset(*B, 0, size);

    cuda_malloc_check((void **)C, size);
    cudaMemset(*C, 0, size);

    cuda_malloc_check((void **)D, size);
    cudaMemset(*D, 0, size);

    cuda_malloc_check((void **)E, size);
    cudaMemset(*E, 0, size);

    cuda_malloc_check((void **)F, size);
    cudaMemset(*F, 0, size);
}

// Utility function for freeing memory of 6 arrays
inline void cuda_free_6_arrays(float** A, float** B, float** C, float** D, float** E, float** F)
{
    cudaFree(*A); *A = NULL;
    cudaFree(*B); *B = NULL;
    cudaFree(*C); *C = NULL;
    cudaFree(*D); *D = NULL;
    cudaFree(*E); *E = NULL;
    cudaFree(*F); *F = NULL;
}


// Multiple TLSs inside FDTD, with TF-IF box
void multiple_TLS_ring_resonator_w_box(bool get_background)
{
    // Part 1: prepare by calculating simulation
    // Constants
    const float um = 1.0f;

    // Wavelength basics
    float SOURCE_WAVELENGTH = 1.55 * um;
    float SOURCE_FREQUENCY = c0 / SOURCE_WAVELENGTH;  // Frequency of the source
    float SOURCE_OMEGA = 2 * PI * SOURCE_FREQUENCY;
    printf("Frequency of source: %.4f Hz \n", SOURCE_FREQUENCY);

    // Materials: Si (waveguide + ring) and SiO2 (substrate)
    float n_Si = 3.4777;
    float n_SiO2 = 1.444;

    Complex eps_air = Complex(1.0f, 0.0f);
    Complex eps_Si = Complex(n_Si * n_Si, 0.0f);
    Complex eps_SiO2 = Complex(n_SiO2 * n_SiO2, 0.0f);

    // Geometry of photonic structure
    float ring_radius = 3 * um;  // radius: 5 um?
    float wg_width = 0.48 * um;
    float wg_height = 0.28 * um;

    float couple_spacing = 0.08 * um;  // Distance between cavity and waveguide

    // Print the results (optional)
    printf("Ring resonator parameters: \n");
    printf("Waveguide width: %f um\n", wg_width);
    printf("Radius: %f um\n", ring_radius);
    printf("Spacing: %f um\n", couple_spacing);

    // Parameters: TLS
    float d0 = 1e-2;  // Dipole moment
    float freq_TLS = 0.65265;
    float omega_TLS = 2 * PI * freq_TLS;  // TLS's resonance frequency
    printf("Frequency of TLS: %.4f Hz \n", freq_TLS);
    // Frequency will shift due to self-interaction
    
    // Decay rate inside vacuum
    float Gamma0 = pow(d0, 2.0) * pow(omega_TLS, 3.0) / (3.0f * PI * hbar * eps0 * pow(c0, 3.0));
    printf("TLS decay rate: %.7f \n", Gamma0);

    // Grid resolution dx
    float dx = 0.04f * um;
    // Time step interval dt
    float dt = 0.56f * dx / c0;  // Courant factor: c * dt < dx / sqrt(3)
    
    int max_iter = 800000;  // Number of time-steps: 800000
    printf("Total time-steps: %d \n", max_iter);
    printf("Decay total time-steps: %.2f \n", (5.0 / Gamma0) / dt);

    // Domain size parameters
    float h_PML = 1.2f * um;  // Thickness of PML
    int t_PML = ceil(h_PML / dx);
    int t_PML_rad = 15;

    float boundary_spacing = h_PML + 0.6 * um;

    float Lx = 2 * ring_radius + 2 * boundary_spacing;
    float Ly = 2 * ring_radius + 1 * wg_width + 1 * couple_spacing + 2 * boundary_spacing;
    float Lz = 2 * boundary_spacing;
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

    // Now we need to define the photonic structures used
    // Initialize mask
    bool *mask;
    mask = (bool *)malloc(Nx * Ny * sizeof(bool));
    memset(mask, 0, Nx * Ny * sizeof(bool));

    // Now let's construct the ring resonator!
    if (!get_background)
        create_ring_resonator_in_mask(mask, Lx, Ly, Nx, Ny, dx, 0.0, 0.0, ring_radius, wg_width);

    // Now let's construct the waveguide
    // Calculate the k indices for structure
    int wg_height_grid = round(wg_height / dx);
    int wg_k_min = k_mid - wg_height_grid / 2;
    int wg_k_max = wg_k_min + wg_height_grid;
    printf("Structure k range: [%d, %d] \n", wg_k_min, wg_k_max);

    // Add a rectangle
    float wg_xmin = -Lx;
    float wg_xmax = Lx;
    float wg_ymin = ring_radius + wg_width / 2.0 + couple_spacing;
    float wg_ymax = wg_ymin + wg_width;

    std::vector<Point2D> wg_vertices = {
        {wg_xmin, wg_ymin},
        {wg_xmax, wg_ymin},
        {wg_xmax, wg_ymax},
        {wg_xmin, wg_ymax}};
    add_polygon_to_mask(mask, wg_vertices, Lx, Ly, Nx, Ny, dx, true);
    // The structure has been written inside "mask".

    char mask_filename[50];
    snprintf(mask_filename, sizeof(mask_filename), "figures/mask.png");
    save_mask_png(mask, mask_filename, Nx, Ny);

    // We need to define auxialiary FDTDs of radiation field
    int box_half = 1;  // Box size: 2 * box_half + 1
    int N_rad_half = box_half + t_PML_rad + ceil(0.3 * SOURCE_WAVELENGTH / dx);
    int N_rad = 2 * N_rad_half + 1;
    int N_rad_mid = N_rad_half + 1;  // The dipole position: at the middle
    printf("Extra FDTD size: (%d, %d, %d) \n", N_rad, N_rad, N_rad);
    
    // TLS placement
    // We arrange them in an array!
    int row_TLS = 3;
    int col_TLS = 2;
    int N_TLS = row_TLS * col_TLS;  // Total number of TLSs in the array

    float neighbor_dist = 2 * dx;

    float Lx_TLS_arr = (row_TLS - 1) * neighbor_dist;
    float Ly_TLS_arr = (col_TLS - 1) * neighbor_dist;

    // The corner of the TLS array
    float x_min_TLS = -Lx_TLS_arr / 2.0 + ring_radius - dx;  // Offset
    float y_min_TLS = -Ly_TLS_arr / 2.0;

    // Position of the TLS
    std::vector<int> i_TLS_arr, j_TLS_arr, k_TLS_arr;
    std::vector<int> idx_TLS_arr;

    // Indices for box boundaries [i1, i2] x [j1, j2] x [k1, k2]
    std::vector<int> i1_TLS_arr, j1_TLS_arr, k1_TLS_arr;
    std::vector<int> i2_TLS_arr, j2_TLS_arr, k2_TLS_arr;

    std::vector<Complex> b_arr;

    for (int row=0; row < row_TLS; ++row)
    {
        for (int col=0; col < col_TLS; ++col)
        {
            float x_TLS = x_min_TLS + row * neighbor_dist;
            float y_TLS = y_min_TLS + col * neighbor_dist;

            int i_TLS = i_mid + round(x_TLS / dx);
            int j_TLS = j_mid + round(y_TLS / dx);
            int k_TLS = k_mid;

            i_TLS_arr.push_back(i_TLS);
            j_TLS_arr.push_back(j_TLS);
            k_TLS_arr.push_back(k_TLS);

            // Indices for box boundaries [i1, i2] x [j1, j2] x [k1, k2]
            i1_TLS_arr.push_back(i_TLS - box_half);
            i2_TLS_arr.push_back(i_TLS + box_half);
            j1_TLS_arr.push_back(j_TLS - box_half);
            j2_TLS_arr.push_back(j_TLS + box_half);
            k1_TLS_arr.push_back(k_TLS - box_half);
            k2_TLS_arr.push_back(k_TLS + box_half);

            // Indices in 3D array
            idx_TLS_arr.push_back(i_TLS + j_TLS * Nx + k_TLS * Nx * Ny);
            
            printf("%d-th TLS located at pixel (%d, %d, %d) \n", 
                row * col_TLS + col, i_TLS, j_TLS, k_TLS);

            float b_init_value = 1.0 / sqrtf(N_TLS);
            b_arr.push_back(Complex(0.0, 0.0));  // Start from ground state
        }
    }
    // The indices in auxiliary FDTDs are the same for all TLSs
    // so there's no need to save these indices in an extra array!
    int idx_TLS_rad = N_rad_mid + N_rad_mid * N_rad + N_rad_mid * (N_rad * N_rad);
    
    int i1_rad = N_rad_mid - box_half; int i2_rad = N_rad_mid + box_half;
    int j1_rad = N_rad_mid - box_half; int j2_rad = N_rad_mid + box_half;
    int k1_rad = N_rad_mid - box_half; int k2_rad = N_rad_mid + box_half;

    // Part 2: initialize arrays and allocate memory for main FDTD
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
    cuda_malloc_memset_6_arrays(&Ex, &Ey, &Ez, &Hx, &Hy, &Hz, size);

    // Allocate memory for the J and M sources on the device
    cuda_malloc_memset_6_arrays(&Jx, &Jy, &Jz, &Mx, &My, &Mz, size);

    // Allocate memory for the C matrices on the device
    cuda_malloc_memset_6_arrays(&Cax, &Cbx, &Cay, &Cby, &Caz, &Cbz, size);
    
    // Allocate memory for the D matrices on the device
    cuda_malloc_memset_6_arrays(&Dax, &Dbx, &Day, &Dby, &Daz, &Dbz, size);

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
    set_FDTD_matrices_3D_structure(Cax_host, Cbx_host, Cay_host, Cby_host, Caz_host, Cbz_host, 
        Dax_host, Dbx_host, Day_host, Dby_host, Daz_host, Dbz_host, 
        Nx, Ny, Nz, dx, dt, mask, eps_air, eps_Si, eps_SiO2, 
        wg_k_min, wg_k_max, SOURCE_OMEGA, t_PML);
    printf("Completed!\n");

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

    // Part 3: initialize arrays and allocate memory for auxiliary FDTDs
    // Total size of the 3D array
    size_t size_rad = N_rad * N_rad * N_rad * sizeof(float);
    
    // Initialize the radiation field
    // E field
    float **Ex_rad, **Ey_rad, **Ez_rad;
    Ex_rad = (float **)malloc(N_TLS * sizeof(float *));
    Ey_rad = (float **)malloc(N_TLS * sizeof(float *));
    Ez_rad = (float **)malloc(N_TLS * sizeof(float *));
    // H field
    float **Hx_rad, **Hy_rad, **Hz_rad;
    Hx_rad = (float **)malloc(N_TLS * sizeof(float *));
    Hy_rad = (float **)malloc(N_TLS * sizeof(float *));
    Hz_rad = (float **)malloc(N_TLS * sizeof(float *));

    // Allocate device memory for radiation field of each TLS
    for (int i = 0; i < N_TLS; ++i) 
    {
        cuda_malloc_memset_6_arrays(&Ex_rad[i], &Ey_rad[i], &Ez_rad[i], &Hx_rad[i], &Hy_rad[i], &Hz_rad[i], size_rad);
    }

    // Initialize the current sources for auxiliary FDTDs
    // J source
    float **Jx_rad, **Jy_rad, **Jz_rad;
    Jx_rad = (float **)malloc(N_TLS * sizeof(float *));
    Jy_rad = (float **)malloc(N_TLS * sizeof(float *));
    Jz_rad = (float **)malloc(N_TLS * sizeof(float *));

    // M source
    float **Mx_rad, **My_rad, **Mz_rad;
    Mx_rad = (float **)malloc(N_TLS * sizeof(float *));
    My_rad = (float **)malloc(N_TLS * sizeof(float *));
    Mz_rad = (float **)malloc(N_TLS * sizeof(float *));

    // Allocate device memory for current sources of each TLS
    for (int i = 0; i < N_TLS; ++i)
    {
        cuda_malloc_memset_6_arrays(&Jx_rad[i], &Jy_rad[i], &Jz_rad[i], &Mx_rad[i], &My_rad[i], &Mz_rad[i], size_rad);
    }

    float *Cax_rad, *Cbx_rad, *Cay_rad, *Cby_rad, *Caz_rad, *Cbz_rad; // On device
    float *Dax_rad, *Dbx_rad, *Day_rad, *Dby_rad, *Daz_rad, *Dbz_rad; // On device

    // Allocate memory for the C matrices on the device
    cuda_malloc_memset_6_arrays(&Cax_rad, &Cbx_rad, &Cay_rad, &Cby_rad, &Caz_rad, &Cbz_rad, size_rad);

    // Allocate memory for the D matrices on the device
    cuda_malloc_memset_6_arrays(&Dax_rad, &Dbx_rad, &Day_rad, &Dby_rad, &Daz_rad, &Dbz_rad, size_rad);

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

    printf("Pre-process: setting permittivity for auxiliary FDTDs...\n");
    set_FDTD_matrices_3D_homo(Cax_rad_host, Cbx_rad_host, Cay_rad_host, Cby_rad_host, 
        Caz_rad_host, Cbz_rad_host, Dax_rad_host, Dbx_rad_host, Day_rad_host, 
        Dby_rad_host, Daz_rad_host, Dbz_rad_host, 
        N_rad, N_rad, N_rad, dx, dt, eps_Si, SOURCE_OMEGA, t_PML_rad);

    // Copy C & D matrices to device
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

    // Part 4: define the incident wave
    // Construct the source: time profile of Gaussian pulse
    float freq_sigma = 0.1 * SOURCE_FREQUENCY;
    float t_sigma = 1 / freq_sigma / (2 * PI); // Used to calculate Gaussian pulse width
    float t_peak = 5 * t_sigma;
    printf("Source total time-steps: %.2f \n", (t_peak + 5 * t_sigma) / dt);
    
    // Incident wave
    float Amp_inc_wave = 0.2f;

    float x_sigma = c0 * t_sigma;
    float xc_init = -5 * x_sigma; // initial center of wave packet

    // Which waveguide mode do you want to excite?
    // We use the fundamental TE mode
    float beta = 2.562927 * SOURCE_OMEGA / c0;
    float phase_velocity = SOURCE_OMEGA / beta;

    // Read in data for waveguide mode profile
    printf("Reading waveguide mode profile...\n");
    int Ny_wg_mode, Nz_wg_mode;

    // Read in E fields, real & imag part
    std::vector<std::vector<float>> ReEy_wg_mode_host = read_2D_array_from_file("data/wg_mode/ReEy.txt", 
            &Ny_wg_mode, &Nz_wg_mode);
    std::vector<std::vector<float>> ImEy_wg_mode_host = read_2D_array_from_file("data/wg_mode/ImEy.txt", 
            &Ny_wg_mode, &Nz_wg_mode);
    std::vector<std::vector<float>> ReEz_wg_mode_host = read_2D_array_from_file("data/wg_mode/ReEz.txt", 
            &Ny_wg_mode, &Nz_wg_mode);
    std::vector<std::vector<float>> ImEz_wg_mode_host = read_2D_array_from_file("data/wg_mode/ImEz.txt", 
            &Ny_wg_mode, &Nz_wg_mode);

    // Read in H fields, real & imag part
    std::vector<std::vector<float>> ReHy_wg_mode_host = read_2D_array_from_file("data/wg_mode/ReHy.txt", 
            &Ny_wg_mode, &Nz_wg_mode);
    std::vector<std::vector<float>> ImHy_wg_mode_host = read_2D_array_from_file("data/wg_mode/ImHy.txt", 
            &Ny_wg_mode, &Nz_wg_mode);
    std::vector<std::vector<float>> ReHz_wg_mode_host = read_2D_array_from_file("data/wg_mode/ReHz.txt", 
            &Ny_wg_mode, &Nz_wg_mode);
    std::vector<std::vector<float>> ImHz_wg_mode_host = read_2D_array_from_file("data/wg_mode/ImHz.txt", 
            &Ny_wg_mode, &Nz_wg_mode);

    // Save the above arrays on GPU
    float* ReEy_wg_mode = copyToCudaDevice(ReEy_wg_mode_host, Ny_wg_mode, Nz_wg_mode);
    float* ImEy_wg_mode = copyToCudaDevice(ImEy_wg_mode_host, Ny_wg_mode, Nz_wg_mode);
    float* ReEz_wg_mode = copyToCudaDevice(ReEz_wg_mode_host, Ny_wg_mode, Nz_wg_mode);
    float* ImEz_wg_mode = copyToCudaDevice(ImEz_wg_mode_host, Ny_wg_mode, Nz_wg_mode);

    float* ReHy_wg_mode = copyToCudaDevice(ReHy_wg_mode_host, Ny_wg_mode, Nz_wg_mode);
    float* ImHy_wg_mode = copyToCudaDevice(ImHy_wg_mode_host, Ny_wg_mode, Nz_wg_mode);
    float* ReHz_wg_mode = copyToCudaDevice(ReHz_wg_mode_host, Ny_wg_mode, Nz_wg_mode);
    float* ImHz_wg_mode = copyToCudaDevice(ImHz_wg_mode_host, Ny_wg_mode, Nz_wg_mode);

    // These mode profiles have been obtained using Tidy3d!
    printf("Waveguide mode profile size: (%d, %d) \n", Ny_wg_mode, Nz_wg_mode);

    // Distance of input port, compared to the middle
    float port_x_shift = ring_radius;
    int port_i_shift = round(port_x_shift / dx);

    // Indices for incident mode source
    int i_inc = i_mid - port_i_shift;

    // y direction: centered at the waveguide's center
    int j_inc_mid = j_mid + round((ring_radius + couple_spacing + wg_width) / dx);
    int j_inc_start = j_inc_mid - round(Ny_wg_mode / 2);
    int j_inc_end = j_inc_start + Ny_wg_mode - 1;

    // z direction: centered at the waveguide's center
    int k_inc_mid = k_mid;
    int k_inc_start = k_inc_mid - round(Nz_wg_mode / 2);
    int k_inc_end = k_inc_mid + Nz_wg_mode;
    // Tune this part a little bit in case multiple modes are excited.

    // Part 5: add monitors to save results
    // Allocate memory for field monitors
    float *E_monitor_xy, *E_rad_monitor_xy;
    E_monitor_xy = (float *)malloc(Nx * Ny * sizeof(float));
    E_rad_monitor_xy = (float *)malloc(N_rad * N_rad * sizeof(float));

    // We also need a monitor to measure output power
    // Indices for incident mode source
    int i_power = i_mid + port_i_shift;

    // y direction: centered at the waveguide's center
    int j_power_start = j_inc_mid - 8;
    int j_power_end = j_power_start + 17;  // 17

    // z direction: centered at the waveguide's center
    int k_power_mid = k_inc_mid;
    int k_power_start = k_power_mid - 5;
    int k_power_end = k_power_start + 11;  // 11

    // Prepare for saving fields
    int save_len = (j_power_end - j_power_start + 1) * (k_power_end - k_power_start + 1);

    // Allocate for monitor arrays that will be used for post-processing
    float *Ey_output_monitor = (float *)malloc(max_iter * save_len * sizeof(float));
    float *Ez_output_monitor = (float *)malloc(max_iter * save_len * sizeof(float));

    float *Hy_output_monitor = (float *)malloc(max_iter * save_len * sizeof(float));
    float *Hz_output_monitor = (float *)malloc(max_iter * save_len * sizeof(float));

    if (Hy_output_monitor == NULL || Hz_output_monitor == NULL || 
        Ey_output_monitor == NULL || Ez_output_monitor == NULL) 
    {
        fprintf(stderr, "Failed to allocate memory for monitors.\n");
        exit(EXIT_FAILURE);  // Exit program with a failure status code
    }

    // Also initialize an array on GPU to save field slices
    float *tmp_monitor_GPU;
    cudaMalloc(&tmp_monitor_GPU, save_len * sizeof(float));

    // Initialize an array to save b value
    // Using vector of vectors for 2D array
    std::vector<std::vector<Complex>> b_save_arr(N_TLS, std::vector<Complex>(max_iter));
    
    float tmp_E_drive = 0.0;  // To save Ez field at TLS position
    
    // Visualization settings
    float plot_field_bound = 0.05;  // Plot figure: colorbar bound: 0.1
    int plot_interval = 2500;  // Plot & record fields periodically

    // Thread & block settings
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeRad((N_rad + blockSize.x - 1) / blockSize.x, (N_rad + blockSize.y - 1) / blockSize.y);
    dim3 gridSizeMonitor((j_power_end - j_power_start + blockSize.x) / blockSize.x, 
        (k_power_end - k_power_start + blockSize.y) / blockSize.y);

    // Part 6: Start working on the main-loop
    // Initialize time for profiling
    printf("Starting the main-loop...\n");

    clock_t start, end;
    start = clock();

    // Start the main-loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        float t = iter * dt;
        float pulse_xc = xc_init + phase_velocity * t;  // center of wave packet
        float source_amp = Amp_inc_wave * exp(-pow(-ring_radius - pulse_xc, 2.0f) / (2 * x_sigma * x_sigma));
        
        // Current source for waveguide mode injection:
        // Mode injection in +x direction: n = (1, 0, 0)
        // J = (0, -Hz, Hy), M = (0, Ez, -Ey);
        
        // We will now carry out the 8 steps one-by-one!
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            // Step 1: calculate J_{i}^{n} from b_{i}^{n}
            float J_value = 2 * d0 * omega_TLS * b_arr[id_TLS].imag / pow(dx, 3);
            // Dipole moment: x direction
            cudaMemcpy(Jx_rad[id_TLS] + idx_TLS_rad, &J_value, sizeof(float), cudaMemcpyHostToDevice);
            
            // Step 2: update E_{i} based on J_{i}^{n}
            updateE<<<gridSizeRad, blockSize>>>(Ex_rad[id_TLS], Ey_rad[id_TLS], Ez_rad[id_TLS], 
                Hx_rad[id_TLS], Hy_rad[id_TLS], Hz_rad[id_TLS], Cax_rad, Cbx_rad, Cay_rad, Cby_rad, 
                Caz_rad, Cbz_rad, Jx_rad[id_TLS], Jy_rad[id_TLS], Jz_rad[id_TLS], dx, N_rad, N_rad, N_rad);
            
            cudaDeviceSynchronize();
        }
        
        // Step 3: calculate J_{i, rad}^{n} based on H_{i}^{n}
        // Equivalence principle for J
        // Set Jx, Jy, Jz
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            int i1 = i1_TLS_arr[id_TLS], i2 = i2_TLS_arr[id_TLS];
            int j1 = j1_TLS_arr[id_TLS], j2 = j2_TLS_arr[id_TLS];
            int k1 = k1_TLS_arr[id_TLS], k2 = k2_TLS_arr[id_TLS];

            // Face 1: i = i1
            // Jy[i1, j, k] += Hz_rad[i1-1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
            // Jz[i1, j, k] += -Hy_rad[i1-1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
            // Face 2: i = i2
            // Jy[i2, j, k] += -Hz_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
            // Jz[i2, j, k] += Hy_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
            set_J_rad_x_dir<<<gridSizeRad, blockSize>>>(Hx_rad[id_TLS], Hy_rad[id_TLS], Hz_rad[id_TLS], 
                Jx, Jy, Jz, dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
                j1_rad, j2_rad, k1_rad, k2_rad);
            cudaDeviceSynchronize();
            
            // Face 3: j = j1
            // Jx[i, j1, k] += -Hz_rad[i, j1-1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
            // Jz[i, j1, k] += Hx_rad[i, j1-1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
            // Face 4: j = j2
            // Jx[i, j2, k] += Hz_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
            // Jz[i, j2, k] += -Hx_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
            set_J_rad_y_dir<<<gridSizeRad, blockSize>>>(Hx_rad[id_TLS], Hy_rad[id_TLS], Hz_rad[id_TLS], 
                Jx, Jy, Jz, dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
                j1_rad, j2_rad, k1_rad, k2_rad);
            cudaDeviceSynchronize();
            
            // Face 5: k = k1
            // Jx[i, j, k1] += Hy_rad[i, j, k1-1] / dx for i in (i1, i2-1) and j in (j1, j2)
            // Jy[i, j, k1] += -Hx_rad[i, j, k1-1] / dx for i in (i1, i2) and j in (j1, j2-1)
            // Face 6: k = k2
            // Jx[i, j, k2] += -Hy_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
            // Jy[i, j, k2] += Hx_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
            set_J_rad_z_dir<<<gridSizeRad, blockSize>>>(Hx_rad[id_TLS], Hy_rad[id_TLS], Hz_rad[id_TLS], 
                Jx, Jy, Jz, dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
                j1_rad, j2_rad, k1_rad, k2_rad);
            cudaDeviceSynchronize();
        }
        
        // Step 4: update E based on J_{i, rad}^{n}
        /*
        // Original way: without mode injection
        updateE<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, 
            Jx, Jy, Jz, dx, Nx, Ny, Nz);
        */
        updateE_mode<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Cax, Cbx, Cay, Cby, Caz, Cbz, Jx, Jy, Jz, dx, Nx, Ny, Nz, 
            i_inc, j_inc_start, j_inc_end, k_inc_start, k_inc_end, 
            ReHy_wg_mode, ImHy_wg_mode, ReHz_wg_mode, ImHz_wg_mode, 
            Nz_wg_mode, source_amp, SOURCE_OMEGA, t);
        cudaDeviceSynchronize();
        // Don't forget to reset the current sources to all 0's!S
        cudaMemset(Jx, 0, size);
        cudaMemset(Jy, 0, size);
        cudaMemset(Jz, 0, size);

        // Step 5: calculate M_{i, rad}^{n+1/2} based on E_{i}^{n+1/2}
        // Equivalence principle for M
        // Set Mx, My and Mz
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            int i1 = i1_TLS_arr[id_TLS], i2 = i2_TLS_arr[id_TLS];
            int j1 = j1_TLS_arr[id_TLS], j2 = j2_TLS_arr[id_TLS];
            int k1 = k1_TLS_arr[id_TLS], k2 = k2_TLS_arr[id_TLS];
            // Face 1: i = i1
            // My[i1-1, j, k] += -Ez_rad[i1, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
            // Mz[i1-1, j, k] += Ey_rad[i1, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
            // Face 2: i = i2
            // My[i2, j, k] += Ez_rad[i2, j, k] / dx for j in (j1, j2) and k in (k1, k2-1)
            // Mz[i2, j, k] += -Ey_rad[i2, j, k] / dx for j in (j1, j2-1) and k in (k1, k2)
            set_M_rad_x_dir<<<gridSizeRad, blockSize>>>(Ex_rad[id_TLS], Ey_rad[id_TLS], Ez_rad[id_TLS], 
                Mx, My, Mz, dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
                j1_rad, j2_rad, k1_rad, k2_rad);
            cudaDeviceSynchronize();
            
            // Mx[i, j1-1, k] += Ez_rad[i, j1, k] / dx for i in (i1, i2) and k in (k1, k2-1)
            // Mz[i, j1-1, k] += -Ex_rad[i, j1, k] / dx for i in (i1, i2-1) and k in (k1, k2)
            // Face 4: j = j2
            // Mx[i, j2, k] += -Ez_rad[i, j2, k] / dx for i in (i1, i2) and k in (k1, k2-1)
            // Mz[i, j2, k] += Ex_rad[i, j2, k] / dx for i in (i1, i2-1) and k in (k1, k2)
            set_M_rad_y_dir<<<gridSizeRad, blockSize>>>(Ex_rad[id_TLS], Ey_rad[id_TLS], Ez_rad[id_TLS], 
                Mx, My, Mz, dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
                j1_rad, j2_rad, k1_rad, k2_rad);
            cudaDeviceSynchronize();
            
            // Face 5: k = k1
            // Mx[i, j, k1-1] += -Ey_rad[i, j, k1] / dx for i in (i1, i2) and j in (j1, j2-1)
            // My[i, j, k1-1] += Ex_rad[i, j, k1] / dx for i in (i1, i2-1) and j in (j1, j2)
            // Face 6: k = k2
            // Mx[i, j, k2] += Ey_rad[i, j, k2] / dx for i in (i1, i2) and j in (j1, j2-1)
            // My[i, j, k2] += -Ex_rad[i, j, k2] / dx for i in (i1, i2-1) and j in (j1, j2)
            set_M_rad_z_dir<<<gridSizeRad, blockSize>>>(Ex_rad[id_TLS], Ey_rad[id_TLS], Ez_rad[id_TLS], 
                Mx, My, Mz, dx, Nx, Ny, Nz, N_rad, i1, i2, j1, j2, k1, k2, i1_rad, i2_rad, 
                j1_rad, j2_rad, k1_rad, k2_rad);
            cudaDeviceSynchronize();
        }
            
        // Step 6: update H based on M_{i, rad}^{n+1/2}
        /*
        // Original way, without mode injection
        updateH<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, 
            Mx, My, Mz, dx, Nx, Ny, Nz);
        */
        updateH_mode<<<gridSize, blockSize>>>(Ex, Ey, Ez, Hx, Hy, Hz, 
            Dax, Dbx, Day, Dby, Daz, Dbz, Mx, My, Mz, dx, Nx, Ny, Nz, 
            i_inc, j_inc_start, j_inc_end, k_inc_start, k_inc_end, 
            ReEy_wg_mode, ImEy_wg_mode, ReEz_wg_mode, ImEz_wg_mode, 
            Nz_wg_mode, source_amp, SOURCE_OMEGA, t + 1 * dt / 2.0);
        cudaDeviceSynchronize();
        // Don't forget to reset the current sources to all 0's!S
        cudaMemset(Mx, 0, size);
        cudaMemset(My, 0, size);
        cudaMemset(Mz, 0, size);

        // Step 7: update H_{i} based on H_{i}^{n}
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            updateH<<<gridSizeRad, blockSize>>>(Ex_rad[id_TLS], Ey_rad[id_TLS], Ez_rad[id_TLS], 
                Hx_rad[id_TLS], Hy_rad[id_TLS], Hz_rad[id_TLS], Dax_rad, Dbx_rad, Day_rad, 
                Dby_rad, Daz_rad, Dbz_rad, Mx_rad[id_TLS], My_rad[id_TLS], Mz_rad[id_TLS], 
                dx, N_rad, N_rad, N_rad);
            cudaDeviceSynchronize();
        }

        // Step 8: update b_i for all TLSs
        for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
        {
            // Get the field: copy E field back to host, prepare to drive b
            cudaMemcpy(&tmp_E_drive, Ex + idx_TLS_arr[id_TLS], sizeof(float), cudaMemcpyDeviceToHost);
            
            // Solve ODE and update b
            // Time interval: [t, t + dt]
            int ode_steps = 5;
            Complex b_new = b_arr[id_TLS];
            for (int i_step = 0; i_step < ode_steps; ++i_step)
                b_new = RK4_step(t, b_new, dt / ode_steps, omega_TLS, d0, n_Si * Gamma0, tmp_E_drive, true);
                
            // Record result
            b_save_arr[id_TLS][iter] = b_new;
            b_arr[id_TLS] = b_new;  // Update
        }

        // Field monitor: update if needed
        extract_field_slice<<<gridSizeMonitor, blockSize>>>(Ey, tmp_monitor_GPU, 
            i_power, j_power_start, j_power_end, k_power_start, k_power_end, Nx, Ny);
        cudaMemcpy(Ey_output_monitor + iter * save_len, tmp_monitor_GPU, save_len * sizeof(float), 
            cudaMemcpyDeviceToHost);  // Copy the extracted Ey to host memory
        
        extract_field_slice<<<gridSizeMonitor, blockSize>>>(Ez, tmp_monitor_GPU, 
            i_power, j_power_start, j_power_end, k_power_start, k_power_end, Nx, Ny);
        cudaMemcpy(Ez_output_monitor + iter * save_len, tmp_monitor_GPU, save_len * sizeof(float), 
            cudaMemcpyDeviceToHost);  // Copy the extracted Ez to host memory
        
        extract_field_slice<<<gridSizeMonitor, blockSize>>>(Hy, tmp_monitor_GPU, 
            i_power, j_power_start, j_power_end, k_power_start, k_power_end, Nx, Ny);
        cudaMemcpy(Hy_output_monitor + iter * save_len, tmp_monitor_GPU, save_len * sizeof(float), 
            cudaMemcpyDeviceToHost);  // Copy the extracted Hy to host memory
        
        extract_field_slice<<<gridSizeMonitor, blockSize>>>(Hz, tmp_monitor_GPU, 
            i_power, j_power_start, j_power_end, k_power_start, k_power_end, Nx, Ny);
        cudaMemcpy(Hz_output_monitor + iter * save_len, tmp_monitor_GPU, save_len * sizeof(float), 
            cudaMemcpyDeviceToHost);  // Copy the extracted Hz to host memory
        
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
            
            snprintf(field_filename, sizeof(field_filename), "figures/gpu0/Ex_%04d.png", iter);
            save_field_png(E_monitor_xy, field_filename, Nx, Ny, plot_field_bound);

            // This time we still record E_rad field. 
            // Record: E_rad field at z=0 plane
            size_t slice_pitch_rad = N_rad * sizeof(float); // The size in bytes of the 2D slice row
            k = N_rad_mid;  // Assuming you want the middle slice

            for (int j = 0; j < N_rad; ++j)
            {
                float* device_ptr = Ex_rad[0] + j * N_rad + k * N_rad * N_rad; // Pointer to the start of the row in the desired slice
                float* host_ptr = E_rad_monitor_xy + j * N_rad;  // Pointer to the host memory

                cudaMemcpy(host_ptr, device_ptr, slice_pitch_rad, cudaMemcpyDeviceToHost);
            }
            
            snprintf(field_filename, sizeof(field_filename), "figures/gpu0/Ex_rad_%04d.png", iter);
            save_field_png(E_rad_monitor_xy, field_filename, N_rad, N_rad, plot_field_bound);
        }

        // Stop early
        if (get_background)
        {
            if (iter >= 10000)
                break;
        } 
    }

    end = clock();  // Stop timing, display result
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Program executed in: %.4f seconds\n", cpu_time_used);

    // Free GPU memory
    cuda_free_6_arrays(&Ex, &Ey, &Ez, &Hx, &Hy, &Hz);
    cuda_free_6_arrays(&Jx, &Jy, &Jz, &Mx, &My, &Mz);

    cuda_free_6_arrays(&Cax, &Cbx, &Cay, &Cby, &Caz, &Cbz);
    cuda_free_6_arrays(&Dax, &Dbx, &Day, &Dby, &Daz, &Dbz);

    // Don't forget to free GPU memory for auxiliary FDTDs
    for (int id_TLS = 0; id_TLS < N_TLS; ++id_TLS)
    {
        cuda_free_6_arrays(&Ex_rad[id_TLS], &Ey_rad[id_TLS], &Ez_rad[id_TLS], 
            &Hx_rad[id_TLS], &Hy_rad[id_TLS], &Hz_rad[id_TLS]);

        cuda_free_6_arrays(&Jx_rad[id_TLS], &Jy_rad[id_TLS], &Jz_rad[id_TLS], 
            &Mx_rad[id_TLS], &My_rad[id_TLS], &Mz_rad[id_TLS]);  
    }

    // Don't forget to free GPU memory for monitors
    cudaFree(tmp_monitor_GPU);

    // Also free the pointer of pointers
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

    cuda_free_6_arrays(&Cax_rad, &Cbx_rad, &Cay_rad, &Cby_rad, &Caz_rad, &Cbz_rad);
    cuda_free_6_arrays(&Dax_rad, &Dbx_rad, &Day_rad, &Dby_rad, &Daz_rad, &Dbz_rad);

    // Save the b data for future analysis
    std::ofstream outFile(std::string("data/gpu0/b_N=") + std::to_string(N_TLS) + std::string(".csv"));
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

    // Save the time monitor data for future analysis
    FILE *file_Ey = fopen("data/gpu0/Ey_output_monitor.bin", "wb");
    if (file_Ey != NULL)
    {
        printf("Saving output Ey data: max_iter = %d, save_len = %d \n", max_iter, save_len);
        for (int i = 0; i < max_iter; ++i)
        {
            fwrite(&Ey_output_monitor[i * save_len], sizeof(float), save_len, file_Ey);
        }
        fclose(file_Ey);
    }

    FILE *file_Ez = fopen("data/gpu0/Ez_output_monitor.bin", "wb");
    if (file_Ez != NULL)
    {
        printf("Saving output Ez data: max_iter = %d, save_len = %d \n", max_iter, save_len);
        for (int i = 0; i < max_iter; ++i)
        {
            fwrite(&Ez_output_monitor[i * save_len], sizeof(float), save_len, file_Ez);
        }
        fclose(file_Ez);
    }

    FILE *file_Hy = fopen("data/gpu0/Hy_output_monitor.bin", "wb");
    if (file_Hy != NULL)
    {
        printf("Saving output Hy data: max_iter = %d, save_len = %d \n", max_iter, save_len);
        for (int i = 0; i < max_iter; ++i)
        {
            fwrite(&Hy_output_monitor[i * save_len], sizeof(float), save_len, file_Hy);
        }
        fclose(file_Hy);
    }

    FILE *file_Hz = fopen("data/gpu0/Hz_output_monitor.bin", "wb");
    if (file_Hz != NULL)
    {
        printf("Saving output Hz data: max_iter = %d, save_len = %d \n", max_iter, save_len);
        for (int i = 0; i < max_iter; ++i)
        {
            fwrite(&Hz_output_monitor[i * save_len], sizeof(float), save_len, file_Hz);
        }
        fclose(file_Hz);
    }

    // Free CPU memory
    free(Ey_output_monitor);
    free(Ez_output_monitor);
    free(Hy_output_monitor);
    free(Hz_output_monitor);

    return;
}


int main()
{
    bool get_background = false;

    int gpu_id = 0;  // Change this to the GPU index you want to use
    cudaError_t status = cudaSetDevice(gpu_id);

    if (status != cudaSuccess) 
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));
        return 1;
    }

    // Superradiance: multiple TLSs inside ring resonator
    multiple_TLS_ring_resonator_w_box(get_background);

    return 0;
}
