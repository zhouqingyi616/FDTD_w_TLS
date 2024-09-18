#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>

#include <vector>
#include <iostream>
#include <fstream>

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

// Implement RK4 method to solve Schrodinger equation
std::pair<Complex, Complex> RK4_Schrodinger_step(const Complex& ce, const Complex& cg, float omega_TLS, 
    float d0, float E_drive, float dt) 
{
    auto f_ce = [&](const Complex& ce, const Complex& cg) {
        return Complex(0, -1) * omega_TLS * ce + Complex(0, 1) * (d0 * E_drive / hbar) * cg;
    };

    auto f_cg = [&](const Complex& ce, const Complex& cg) {
        return Complex(0, 1) * (d0 * E_drive / hbar) * ce;
    };

    // Compute k1
    Complex k1_ce = f_ce(ce, cg);
    Complex k1_cg = f_cg(ce, cg);

    // Compute k2
    Complex k2_ce = f_ce(ce + k1_ce * 0.5 * dt, cg + k1_cg * 0.5 * dt);
    Complex k2_cg = f_cg(ce + k1_ce * 0.5 * dt, cg + k1_cg * 0.5 * dt);

    // Compute k3
    Complex k3_ce = f_ce(ce + k2_ce * 0.5 * dt, cg + k2_cg * 0.5 * dt);
    Complex k3_cg = f_cg(ce + k2_ce * 0.5 * dt, cg + k2_cg * 0.5 * dt);

    // Compute k4
    Complex k4_ce = f_ce(ce + k3_ce * dt, cg + k3_cg * dt);
    Complex k4_cg = f_cg(ce + k3_ce * dt, cg + k3_cg * dt);

    // Combine contributions
    Complex ce_new = ce + (k1_ce + k2_ce * 2 + k3_ce * 2 + k4_ce) * dt / 6.0;
    Complex cg_new = cg + (k1_cg + k2_cg * 2 + k3_cg * 2 + k4_cg) * dt / 6.0;

    return std::make_pair(ce_new, cg_new);
}

// Implement RK4 method to solve Bloch equation
std::pair<Complex, Complex> RK4_Bloch_step(Complex rho_ee, Complex rho_eg, float omega_TLS, 
    float d0, float E, float Gamma0, float dt) 
{
    // Derivative functions for Bloch equations
    auto f_rho_ee = [&](Complex rho_ee, Complex rho_eg) {
        return Complex(0, 1) * d0 * E / hbar * (rho_eg.conj() - rho_eg) - rho_ee * Gamma0;
    };

    auto f_rho_eg = [&](Complex rho_ee, Complex rho_eg) {
        return (Complex(0, -1) * omega_TLS - Gamma0 / 2.0f) * rho_eg + 
            Complex(0, 1) * d0 * E / hbar * (rho_ee * (-2.0) + 1.0);
    };

    // Compute k1
    Complex k1_ee = f_rho_ee(rho_ee, rho_eg); // Only real part affects rho_ee
    Complex k1_eg = f_rho_eg(rho_ee, rho_eg);

    // Compute k2
    Complex k2_ee = f_rho_ee(rho_ee + k1_ee * 0.5 * dt, rho_eg + k1_eg * 0.5 * dt);
    Complex k2_eg = f_rho_eg(rho_ee + k1_ee * 0.5 * dt, rho_eg + k1_eg * 0.5 * dt);

    // Compute k3
    Complex k3_ee = f_rho_ee(rho_ee + k2_ee * 0.5 * dt, rho_eg + k2_eg * 0.5 * dt);
    Complex k3_eg = f_rho_eg(rho_ee + k2_ee * 0.5 * dt, rho_eg + k2_eg * 0.5 * dt);

    // Compute k4
    Complex k4_ee = f_rho_ee(rho_ee + k3_ee * dt, rho_eg + k3_eg * dt);
    Complex k4_eg = f_rho_eg(rho_ee + k3_ee * dt, rho_eg + k3_eg * dt);

    // Combine contributions
    Complex rho_ee_new = rho_ee + (k1_ee + k2_ee * 2 + k3_ee * 2 + k4_ee) * dt / 6.0;
    Complex rho_eg_new = rho_eg + (k1_eg + k2_eg * 2 + k3_eg * 2 + k4_eg) * dt / 6.0;

    return std::make_pair(rho_ee_new, rho_eg_new);
}

// Functions for saving data as image
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

// Function for setting C, D matrices used in FDTD
void set_FDTD_matrices_3D(float *Cax, float *Cbx, float *Cay, float *Cby, float *Caz, float *Cbz, 
    float *Dax, float *Dbx, float *Day, float *Dby, float *Daz, float *Dbz, 
    int Nx, int Ny, int Nz, float dx, float dt, Complex eps_air, float OMEGA0, int t_PML)
{
    // Parameters for PML
    float a_max = 2.0;
    int p = 3;
    float eta0 = sqrt(mu0 / eps0);
    float sigma_max = -(p + 1) * std::log(2e-5) / (2 * eta0 * t_PML * dx);
    
    for (int i = 0; i < Nx; ++i) 
    {
        for (int j = 0; j < Ny; ++j) 
        {
            for (int k = 0; k < Nz; ++k)
            {
                // Initialize eps_r and mu_r for this point
                Complex eps_r(1.0, 0.0);
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


// Function for reading array data from file
std::vector<float> read_array_from_file(const char* filename) 
{
    std::vector<float> array;
    std::ifstream file(filename);
    double value;

    while (file >> value)
        array.push_back(value);

    return array;
}

# endif
