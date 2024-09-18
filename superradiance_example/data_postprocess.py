import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter1d


def TLS_dynamics_analyze():
    # Constants
    eps0 = 1.0
    mu0 = 1.0
    c0 = 1 / np.sqrt(eps0 * mu0)
    hbar = 1.0

    # TLS parameters
    d0 = 2e-3
    omega_TLS = 2 * np.pi * 1.0
    Gamma0 = (d0 ** 2) * (omega_TLS ** 3) / (3.0 * np.pi * hbar * eps0 * c0 ** 3)
    # We work in 3D, and dipole is in z direction

    # Time-step
    dx = 0.04
    dt = 0.56 * dx

    # Number of TLSs
    N_TLS = 4

    # Read in b data from file
    data = np.genfromtxt("data/b_N=" + str(N_TLS) + ".csv", dtype='complex', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_iter = data.shape[0]  # Total number of steps
    print(data.shape)
    
    t_arr = np.arange(1, max_iter + 1) * dt

    b_real = data[:, :].real
    b_imag = data[:, :].imag

    b_abs = np.sqrt(b_real ** 2 + b_imag ** 2)

    # Excited probability
    Pe = b_abs ** 2
    P_tot = np.sum(Pe, axis=1)

    # Apply Gaussian smoothing
    sigma = 30  # Smoothing parameter, adjust based on your data's noise level and resolution
    P_tot_smoothed = gaussian_filter1d(P_tot, sigma=sigma)
    
    # Visualize the time-evolution of Pe
    plt.figure(figsize=(10, 6))
    for id_TLS in range(N_TLS):
        plt.plot(t_arr, Pe[:, id_TLS], label=str(id_TLS) + "-th TLS", linewidth=1)

    # plt.xlim([0, 10])
    plt.ylim([0, 1])

    plt.title("Time evolution of excited probability")
    plt.xlabel("Time (sec)")
    plt.ylabel("Pe")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Visualize the decay of total excitation number
    plt.figure(figsize=(10, 6))  # Enhanced figure size

    plt.plot(t_arr, P_tot, label='Total P', color='red', linewidth=1)
    plt.plot(t_arr, P_tot_smoothed, label='smoothed', linewidth=1)
    plt.plot(t_arr, P_tot[0] * np.exp(-Gamma0 * t_arr), label='exp[-Gamma * t]', linewidth=1)
    plt.plot(t_arr, P_tot[0] * np.exp(-N_TLS * Gamma0 * t_arr), label='Super: exp[-N * Gamma * t]', linewidth=1)

    plt.title("TLS: decay of total excited probability")
    plt.xlabel("Time (sec)")
    plt.ylabel("P")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Visualize the time derivative of total excitation number
    # dP_tot = np.diff(P_tot) / dt
    dP_tot = np.diff(P_tot_smoothed) / dt  # Use the smoothed version
    t_derivative = t_arr[:-1] + dt / 2

    # Plot dP_tot/dt
    plt.figure(figsize=(10, 6))
    plt.plot(Gamma0 * t_derivative, -dP_tot / (N_TLS * Gamma0), label='dP/dt', color='blue', linewidth=1)
    plt.title("TLS: Derivative of total excited probability")
    plt.xlabel("Gamma * t")
    plt.ylabel("dP/dt")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
  

if __name__ == "__main__":
    # Plot the time-evolution of TLS
    TLS_dynamics_analyze()


