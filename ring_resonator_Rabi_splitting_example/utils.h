#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PI 3.14159265359f

#define eps0 1.0f
#define mu0 1.0f
#define c0 (1.0f / sqrt(mu0 * eps0))
#define hbar 1.0f

struct Point2D
{
    float x, y;
};

// Class for complex number
class Complex
{
    public:
        float real, imag;

        Complex(): real(0.0), imag(0.0) {}

        Complex(float real_, float imag_): real(real_), imag(imag_) {}

        // Assignment operator
        Complex& operator=(const Complex& other)
        {
            // Handle self-assignment: check if the current object is not the same as the other
            if (this != &other) 
            {
                real = other.real;
                imag = other.imag;
            }
            return *this;  // Return a reference to this object
        }

        // Addition of two complex numbers
        Complex operator+(const Complex& b) const
        {
            return Complex(real + b.real, imag + b.imag);
        }

        // Subtraction of two complex numbers
        Complex operator-(const Complex& b) const
        {
            return Complex(real - b.real, imag - b.imag);
        }

        // Multiplication of two complex numbers
        Complex operator*(const Complex& b) const
        {
            float newReal = real * b.real - imag * b.imag;
            float newImag = real * b.imag + imag * b.real;
            return Complex(newReal, newImag);
        }

        // Division of two complex numbers
        Complex operator/(const Complex& b) const
        {
            float denominator = b.real * b.real + b.imag * b.imag;
            float newReal = (real * b.real + imag * b.imag) / denominator;
            float newImag = (imag * b.real - real * b.imag) / denominator;
            return Complex(newReal, newImag);
        }

        // Addition with float number
        Complex operator+(const float& b) const
        {
            return Complex(real + b, imag);
        }

        // Subtraction with float number
        Complex operator-(const float& b) const
        {
            return Complex(real - b, imag);
        }

        // Multiplication with float number
        Complex operator*(const float& b) const
        {
            return Complex(real * b, imag * b);
        }

        // Division by float number
        Complex operator/(const float& b) const
        {
            return Complex(real / b, imag / b);
        }

        // Complex conjugate
        Complex conj()
        {
            return Complex(real, -imag);
        }

        // Amplitude
        float abs()
        {
            return sqrt(real * real + imag * imag);
        }
};

// Functions for sin() & cos()
float cosd(float angle)
{
    return std::cos(angle * PI / 180.0f);
}

float sind(float angle)
{
    return std::sin(angle * PI / 180.0f);
}


// Functions related to TLS' dynamics
// The ODE of TLS: db/dt = f(t, b) = -1j * omega0 * b - 1j * d0 * E / hbar 
Complex dbdt_original(float t, Complex b, float omega0, float d0, float E_drive)
{
    return Complex(0.0, -1.0) * (b * omega0 - d0 * E_drive / hbar);  // Original version
}

Complex dbdt_TFIF(float t, Complex b, float omega0, float d0, float Gamma0, float E_drive)
{
    return (Complex(0.0, -1.0) * omega0 - Gamma0 / 2) * b + 
        Complex(0.0, 1.0) * d0 * E_drive / hbar;  // Box version
}


// Implement RK4 method to solve ODE
Complex RK4_step(float t, Complex b, float dt, float omega0, float d0, float Gamma0, 
    float E, bool use_box) 
{
    Complex k1, k2, k3, k4;
    if (use_box == false)
    {
        k1 = dbdt_original(t, b, omega0, d0, E) * dt;
        k2 = dbdt_original(t + dt/2, b + k1/2.0, omega0, d0, E) * dt;
        k3 = dbdt_original(t + dt/2, b + k2/2.0, omega0, d0, E)* dt;
        k4 = dbdt_original(t + dt, b + k3, omega0, d0, E) * dt;
    }
    else
    {
        k1 = dbdt_TFIF(t, b, omega0, d0, Gamma0, E) * dt;
        k2 = dbdt_TFIF(t + dt/2, b + k1/2.0, omega0, d0, Gamma0, E) * dt;
        k3 = dbdt_TFIF(t + dt/2, b + k2/2.0, omega0, d0, Gamma0, E)* dt;
        k4 = dbdt_TFIF(t + dt, b + k3, omega0, d0, Gamma0, E) * dt;
    }

    return b + (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
}

// Functions for saving data as image
void save_mask_png(bool *mask, const char filename[], int Nx, int Ny)
{
    // Allocate memory for img in a single block
    unsigned char *img = (unsigned char *)malloc(Nx * Ny * 3 * sizeof(unsigned char));

    if (img == NULL) 
    {
        perror("Failed to allocate memory for img");
        return;
    }

    // Convert the data to image format
    for (int i = 0; i < Nx; i++) 
    {
        for (int j = 0; j < Ny; j++) 
        {
            int idx = (i + j * Nx) * 3;
            if (mask[i + j * Nx]) 
            {
                img[idx] = img[idx + 1] = img[idx + 2] = 0; // Black for mask true
            }
            else
            {
                img[idx] = img[idx + 1] = img[idx + 2] = 255; // White for mask false
            }
        }
    }

    // Write to file
    // if (stbi_write_png(filename, Ny, Nx, 3, img, Ny * 3) == 0)
    if (stbi_write_png(filename, Nx, Ny, 3, img, Nx * 3) == 0)
    {
        // Parameter: width, height
        // So it should be (Ny, Nx) to get a Nx x Ny figure. 
        // The "stride_bytes" parameter should be Ny * 3. 
        perror("Failed to write image");
    }

    // Free the allocated memory
    free(img);
}

void save_field_png(float *u, const char filename[], int Nx, int Ny, float vmax)
{
    // Allocate memory for img in a single block
    unsigned char *img = (unsigned char *)malloc(Nx * Ny * 3 * sizeof(unsigned char));

    if (img == NULL) 
    {
        perror("Failed to allocate memory for img");
        return;
    }

    // Convert the data to image format
    for (int i = 0; i < Nx; i++) 
    {
        for (int j = 0; j < Ny; j++) 
        {
            int idx_field = i + j * Nx;// i * Ny + j;
            int idx_img = idx_field * 3;

            float value = u[idx_field] / vmax;

            unsigned char red, green, blue;

            if (value >= 0)
            {
                if (value > 1)
                    value = 1;

                // Interpolate between white and red
                red = 255;
                green = blue = (unsigned char)(255 * (1 - value)); // Decrease green and blue for positive values
            }
            else
            {
                if (value < -1)
                    value = -1;

                // Interpolate between white and blue
                blue = 255;
                red = green = (unsigned char)(255 * (1 + value)); // Decrease red and green for negative values
            }

            img[idx_img] = red;
            img[idx_img + 1] = green;
            img[idx_img + 2] = blue;
        }
    }

    // Write to file
    if (stbi_write_png(filename, Nx, Ny, 3, img, Nx * 3) == 0) 
    {
        // Parameter: width, height
        // So it should be (Ny, Nx) to get a Nx x Ny figure. 
        // The "stride_bytes" parameter should be Ny * 3. 
        perror("Failed to write image");
    }

    // Free the allocated memory
    free(img);
}

// Geometric utility function
// Judge whether a 2D point is inside a given polygon
bool is_inside_polygon(const Point2D& p, const std::vector<Point2D>& vertices) 
{
    int count = 0;
    int n = vertices.size();
    if (n < 3) return false; // Not a polygon

    for (int i = 0; i < n; i++) 
    {
        Point2D v1 = vertices[i];
        Point2D v2 = vertices[(i + 1) % n]; // Loop back to the start for the last edge

        // Check if the ray intersects with the edge (v1, v2)
        if ((v1.y > p.y) != (v2.y > p.y)) 
        {
            // Compute the x-coordinate of the intersection
            double xIntersect = v1.x + (p.y - v1.y) * (v2.x - v1.x) / (v2.y - v1.y);
            if (p.x < xIntersect)
                count++;
        }
    }

    return count % 2 != 0;  // Odd count means inside
}

// Geometric utility function: add a polygon to mask
void add_polygon_to_mask(bool* mask, const std::vector<Point2D>& vertices, float Lx, float Ly, 
    int Nx, int Ny, float dx, bool value)
{
    // Find the x_min, x_max, y_min, y_max of vertices
    int N = vertices.size();
    float x_min = vertices[0].x;
    float x_max = vertices[0].x;
    float y_min = vertices[0].y;
    float y_max = vertices[0].y;

    for (int i = 1; i < N; i++) 
    {
        if (vertices[i].x < x_min) x_min = vertices[i].x;
        if (vertices[i].x > x_max) x_max = vertices[i].x;
        if (vertices[i].y < y_min) y_min = vertices[i].y;
        if (vertices[i].y > y_max) y_max = vertices[i].y;
    }

    // Find the indices to start and end
    int i_min = max(0, (int) floor((x_min + Lx / 2) / dx));
    int i_max = min(Nx - 1, (int) ceil((x_max + Lx / 2) / dx));

    int j_min = max(0, (int) floor((y_min + Ly / 2) / dx));
    int j_max = min(Ny - 1, (int) ceil((y_max + Ly / 2) / dx));

    // Iterate over grid points in this range
    for (int i = i_min; i < i_max; ++i)
    {
        for (int j = j_min; j < j_max; ++j)
        {
            Point2D p {-Lx/2 + i * dx, -Ly/2 + j * dx};
            if (is_inside_polygon(p, vertices))
                mask[i + j * Nx] = value;  // mask[i * Ny + j] = value;
        }
    }
}

// Geometric utility function: add a circle to mask
void add_circle_to_mask(bool* mask, float xc, float yc, float r, float Lx, float Ly, 
    int Nx, int Ny, float dx, bool value)
{
    float x_min = xc - r;
    float x_max = xc + r;
    float y_min = yc - r;
    float y_max = yc + r;

    // Find the indices to start and end
    int i_min = max(0, (int) floor((x_min + Lx / 2) / dx));
    int i_max = min(Nx - 1, (int) ceil((x_max + Lx / 2) / dx));

    int j_min = max(0, (int) floor((y_min + Ly / 2) / dx));
    int j_max = min(Ny - 1, (int) ceil((y_max + Ly / 2) / dx));

    // Iterate over grid points in this range
    for (int i = i_min; i < i_max; ++i)
    {
        for (int j = j_min; j < j_max; ++j)
        {
            Point2D p {-Lx/2 + i * dx, -Ly/2 + j * dx};
            // Judge: if the point lies inside circle
            if (pow(p.x - xc, 2.0) + pow(p.y - yc, 2.0) < r * r)
            {
                mask[i + j * Nx] = value;
                // printf("%.4f, %.4f \n", p.x, p.y);
            }
        }
    }
}

// Geometric utility function: add a ring resonator
void create_ring_resonator_in_mask(bool* mask, float Lx, float Ly, int Nx, int Ny, float dx, float xc, float yc, 
    float cavity_radius, float wg_width)
{
    // Construct the outer ring
    add_circle_to_mask(mask, xc, yc, cavity_radius + wg_width / 2, Lx, Ly, Nx, Ny, dx, true);
    // Construct the inner ring
    add_circle_to_mask(mask, xc, yc, cavity_radius - wg_width / 2, Lx, Ly, Nx, Ny, dx, false);
}

// Function for setting C, D matrices used in FDTD
void set_FDTD_matrices_3D_homo(float *Cax, float *Cbx, float *Cay, float *Cby, float *Caz, float *Cbz, 
    float *Dax, float *Dbx, float *Day, float *Dby, float *Daz, float *Dbz, 
    int Nx, int Ny, int Nz, float dx, float dt, Complex eps, float OMEGA0, int t_PML)
{
    // Here we consider a homogeneous background with no structure
    // The permittivity is set by input parameter "eps"
    float n_background = sqrt(eps.real);  // Incorrect: background refractive index
    
    // Parameters for PML
    float a_max = 2.0;
    int p = 3;
    float eta0 = sqrt(mu0 / eps0);
    float sigma_max = -(p + 1) * std::log(1e-5) / (2 * eta0 * t_PML * dx);

    for (int i = 0; i < Nx; ++i) 
    {
        for (int j = 0; j < Ny; ++j) 
        {
            for (int k = 0; k < Nz; ++k)
            {
                // Initialize eps_r and mu_r for this point
                Complex eps_r = eps;
                Complex mu_r(1.0, 0.0);

                // Calculate the linear index
                // int idx = i * (Ny * Nz) + j * Nz + k;
                int idx = i + j * Nx + k * Nx * Ny;

                // Calculate sx, sy and sy (for PML)
                Complex sx(1.0, 0.0);
                Complex sy(1.0, 0.0);
                Complex sz(1.0, 0.0);
                
                float bound_x_dist = 0;
                if (i < t_PML)
                {
                    bound_x_dist = 1.0 - float(i) / t_PML;
                }
                if (i + t_PML >= Nx)
                {
                    bound_x_dist = 1.0 - float(Nx - i - 1) / t_PML;
                }
                sx.real = 1 + a_max * std::pow(bound_x_dist, p);
                sx.imag = sigma_max * std::pow(bound_x_dist, p) / (OMEGA0 * eps0);

                float bound_y_dist = 0;
                if (j < t_PML)
                {
                    bound_y_dist = 1.0 - float(j) / t_PML;
                }
                if (j + t_PML >= Ny)
                {
                    bound_y_dist = 1.0 - float(Ny - j - 1) / t_PML;
                }
                sy.real = 1 + a_max * std::pow(bound_y_dist, p);
                sy.imag = sigma_max * std::pow(bound_y_dist, p) / (OMEGA0 * eps0);

                float bound_z_dist = 0;
                if (k < t_PML)
                {
                    bound_z_dist = 1.0 - float(k) / t_PML;
                }
                if (k + t_PML >= Nz)
                {
                    bound_z_dist = 1.0 - float(Nz - k - 1) / t_PML;
                }
                sz.real = 1 + a_max * std::pow(bound_z_dist, p);
                sz.imag = sigma_max * std::pow(bound_z_dist, p) / (OMEGA0 * eps0);

                // Calculate the complex permittivity
                Complex eps_xx_complex = (eps_r * sy * sz) / sx;
                Complex eps_yy_complex = (eps_r * sx * sz) / sy;
                Complex eps_zz_complex = (eps_r * sx * sy) / sz;

                // Calculate the complex permeability
                Complex mu_xx_complex = (mu_r * sy * sz) / sx;
                Complex mu_yy_complex = (mu_r * sx * sz) / sy;
                Complex mu_zz_complex = (mu_r * sx * sy) / sz;

                // Take the real part (used in FDTD)
                float eps_xx = eps_xx_complex.real;
                if (eps_xx < 1) eps_xx = 1;
                float eps_yy = eps_yy_complex.real;
                if (eps_yy < 1) eps_yy = 1;
                float eps_zz = eps_zz_complex.real;
                if (eps_zz < 1) eps_zz = 1;

                float mu_xx = mu_xx_complex.real;
                if (mu_xx < 1) mu_xx = 1;
                float mu_yy = mu_yy_complex.real;
                if (mu_yy < 1) mu_yy = 1;
                float mu_zz = mu_zz_complex.real;
                if (mu_zz < 1) mu_zz = 1;

                // Take the imaginary part
                float sigma_e_xx = std::abs(OMEGA0 * eps0 * eps_xx_complex.imag);
                float sigma_e_yy = std::abs(OMEGA0 * eps0 * eps_yy_complex.imag);
                float sigma_e_zz = std::abs(OMEGA0 * eps0 * eps_zz_complex.imag);

                float sigma_h_xx = std::abs(OMEGA0 * mu0 * mu_xx_complex.imag);
                float sigma_h_yy = std::abs(OMEGA0 * mu0 * mu_yy_complex.imag);
                float sigma_h_zz = std::abs(OMEGA0 * mu0 * mu_zz_complex.imag);

                // Now fill in all the C matrices (used for E field)
                float tmp_x = sigma_e_xx * dt / (2.0 * eps_xx * eps0);

                Cax[idx] = (1.0 - tmp_x) / (1.0 + tmp_x);
                Cbx[idx] = (dt / (eps_xx * eps0)) / (1.0 + tmp_x);

                float tmp_y = sigma_e_yy * dt / (2.0 * eps_yy * eps0);

                Cay[idx] = (1.0 - tmp_y) / (1.0 + tmp_y);
                Cby[idx] = (dt / (eps_yy * eps0)) / (1.0 + tmp_y);

                float tmp_z = sigma_e_zz * dt / (2.0 * eps_zz * eps0);

                Caz[idx] = (1.0 - tmp_z) / (1.0 + tmp_z);
                Cbz[idx] = (dt / (eps_zz * eps0)) / (1.0 + tmp_z);

                // Now fill in all the D matrices (used for H field)
                tmp_x = sigma_h_xx * dt / (2.0 * mu_xx * mu0);

                Dax[idx] = (1.0 - tmp_x) / (1.0 + tmp_x);
                Dbx[idx] = (dt / (mu_xx * mu0)) / (1.0 + tmp_x);

                tmp_y = sigma_h_yy * dt / (2.0 * mu_yy * mu0);

                Day[idx] = (1.0 - tmp_y) / (1.0 + tmp_y);
                Dby[idx] = (dt / (mu_yy * mu0)) / (1.0 + tmp_y);

                tmp_z = sigma_h_zz * dt / (2.0 * mu_zz * mu0);

                Daz[idx] = (1.0 - tmp_z) / (1.0 + tmp_z);
                Dbz[idx] = (dt / (mu_zz * mu0)) / (1.0 + tmp_z);
            }
        }
    }
}


// Function for setting C, D matrices used in FDTD
void set_FDTD_matrices_3D_structure(float *Cax, float *Cbx, float *Cay, float *Cby, float *Caz, float *Cbz, 
    float *Dax, float *Dbx, float *Day, float *Dby, float *Daz, float *Dbz, 
    int Nx, int Ny, int Nz, float dx, float dt, bool* mask,
    Complex eps_air, Complex eps_structure, Complex eps_substrate, 
    int k_min, int k_max, float OMEGA0, int t_PML)
{
    // Parameters for PML
    float a_max = 2.0;
    int p = 3;
    float eta0 = sqrt(mu0 / eps0);
    float sigma_max = -(p + 1) * std::log(1e-5) / (2 * eta0 * t_PML * dx);
    
    printf("(Nx, Ny, Nz): (%d, %d, %d) \n", Nx, Ny, Nz);
    for (int i = 0; i < Nx; ++i) 
    {
        // printf("i: %d \n", i);
        for (int j = 0; j < Ny; ++j) 
        {
            // printf("j: %d \n", j);
            for (int k = 0; k < Nz; ++k)
            {
                // printf("k: %d \n", k);
                // Initialize eps_r and mu_r for this point
                Complex eps_r = eps_air;
                Complex mu_r(1.0, 0.0);

                // Calculate the linear index
                // int idx = i * (Ny * Nz) + j * Nz + k;
                int idx = i + j * Nx + k * Nx * Ny;
                int idx_2D = i + j * Nx;
                
                if (k < k_min)  // Substrate
                    eps_r = eps_substrate;
                else if (k_min <= k && k <= k_max)
                {
                    if (mask[idx_2D])
                        eps_r = eps_structure;
                }
                
                // Calculate sx, sy and sy (for PML)
                Complex sx(1.0, 0.0);
                Complex sy(1.0, 0.0);
                Complex sz(1.0, 0.0);
                
                float bound_x_dist = 0;
                if (i < t_PML)
                {
                    bound_x_dist = 1.0 - float(i) / t_PML;
                }
                if (i + t_PML >= Nx)
                {
                    bound_x_dist = 1.0 - float(Nx - i - 1) / t_PML;
                }
                sx.real = 1 + a_max * std::pow(bound_x_dist, p);
                sx.imag = sigma_max * std::pow(bound_x_dist, p) / (OMEGA0 * eps0);

                float bound_y_dist = 0;
                if (j < t_PML)
                {
                    bound_y_dist = 1.0 - float(j) / t_PML;
                }
                if (j + t_PML >= Ny)
                {
                    bound_y_dist = 1.0 - float(Ny - j - 1) / t_PML;
                }
                sy.real = 1 + a_max * std::pow(bound_y_dist, p);
                sy.imag = sigma_max * std::pow(bound_y_dist, p) / (OMEGA0 * eps0);

                float bound_z_dist = 0;
                if (k < t_PML)
                {
                    bound_z_dist = 1.0 - float(k) / t_PML;
                }
                if (k + t_PML >= Nz)
                {
                    bound_z_dist = 1.0 - float(Nz - k - 1) / t_PML;
                }
                sz.real = 1 + a_max * std::pow(bound_z_dist, p);
                sz.imag = sigma_max * std::pow(bound_z_dist, p) / (OMEGA0 * eps0);

                // Calculate the complex permittivity
                Complex eps_xx_complex = (eps_r * sy * sz) / sx;
                Complex eps_yy_complex = (eps_r * sx * sz) / sy;
                Complex eps_zz_complex = (eps_r * sx * sy) / sz;

                // Calculate the complex permeability
                Complex mu_xx_complex = (mu_r * sy * sz) / sx;
                Complex mu_yy_complex = (mu_r * sx * sz) / sy;
                Complex mu_zz_complex = (mu_r * sx * sy) / sz;

                // Take the real part (used in FDTD)
                float eps_xx = eps_xx_complex.real;
                if (eps_xx < 1) eps_xx = 1;
                float eps_yy = eps_yy_complex.real;
                if (eps_yy < 1) eps_yy = 1;
                float eps_zz = eps_zz_complex.real;
                if (eps_zz < 1) eps_zz = 1;

                float mu_xx = mu_xx_complex.real;
                if (mu_xx < 1) mu_xx = 1;
                float mu_yy = mu_yy_complex.real;
                if (mu_yy < 1) mu_yy = 1;
                float mu_zz = mu_zz_complex.real;
                if (mu_zz < 1) mu_zz = 1;

                // Take the imaginary part
                float sigma_e_xx = std::abs(OMEGA0 * eps0 * eps_xx_complex.imag);
                float sigma_e_yy = std::abs(OMEGA0 * eps0 * eps_yy_complex.imag);
                float sigma_e_zz = std::abs(OMEGA0 * eps0 * eps_zz_complex.imag);

                float sigma_h_xx = std::abs(OMEGA0 * mu0 * mu_xx_complex.imag);
                float sigma_h_yy = std::abs(OMEGA0 * mu0 * mu_yy_complex.imag);
                float sigma_h_zz = std::abs(OMEGA0 * mu0 * mu_zz_complex.imag);

                // Now fill in all the C matrices (used for E field)
                float tmp_x = sigma_e_xx * dt / (2.0 * eps_xx * eps0);

                Cax[idx] = (1.0 - tmp_x) / (1.0 + tmp_x);
                Cbx[idx] = (dt / (eps_xx * eps0)) / (1.0 + tmp_x);

                float tmp_y = sigma_e_yy * dt / (2.0 * eps_yy * eps0);

                Cay[idx] = (1.0 - tmp_y) / (1.0 + tmp_y);
                Cby[idx] = (dt / (eps_yy * eps0)) / (1.0 + tmp_y);

                float tmp_z = sigma_e_zz * dt / (2.0 * eps_zz * eps0);

                Caz[idx] = (1.0 - tmp_z) / (1.0 + tmp_z);
                Cbz[idx] = (dt / (eps_zz * eps0)) / (1.0 + tmp_z);

                // Now fill in all the D matrices (used for H field)
                tmp_x = sigma_h_xx * dt / (2.0 * mu_xx * mu0);

                Dax[idx] = (1.0 - tmp_x) / (1.0 + tmp_x);
                Dbx[idx] = (dt / (mu_xx * mu0)) / (1.0 + tmp_x);

                tmp_y = sigma_h_yy * dt / (2.0 * mu_yy * mu0);

                Day[idx] = (1.0 - tmp_y) / (1.0 + tmp_y);
                Dby[idx] = (dt / (mu_yy * mu0)) / (1.0 + tmp_y);

                tmp_z = sigma_h_zz * dt / (2.0 * mu_zz * mu0);

                Daz[idx] = (1.0 - tmp_z) / (1.0 + tmp_z);
                Dbz[idx] = (dt / (mu_zz * mu0)) / (1.0 + tmp_z);
            }
        }
    }
}


// Function for reading 1D array data from file
std::vector<float> read_1D_array_from_file(const char* filename) 
{
    std::vector<float> array;
    std::ifstream file(filename);
    double value;

    while (file >> value)
        array.push_back(value);

    return array;
}

// Function for reading 2D array data from .txt file
std::vector<std::vector<float>> read_2D_array_from_file(const std::string& filename, int* pNy, int* pNz) 
{
    std::ifstream file(filename);
    if (!file) 
    {
        std::cerr << "Error opening file" << std::endl;
        throw std::runtime_error("Failed to open file.");
    }

    std::vector<std::vector<float>> array;
    std::string line;
    int Ny = 0, Nz = 0;

    // Read lines from the file
    while (getline(file, line)) 
    {
        std::istringstream iss(line);
        float num;
        std::vector<float> temp;

        while (iss >> num)
            temp.push_back(num);

        if (Ny == 0)
            Nz = temp.size(); // Set the number of columns based on the first line

        array.push_back(temp);
        Ny++; // Increment the number of rows
    }

    file.close();

    // Set the dimensions via pointers
    *pNy = Ny;
    *pNz = Nz;

    return array;
}

# endif
