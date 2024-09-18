import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad


def TLS_dynamics_analyze():
    # Constants
    eps0 = 1.0
    mu0 = 1.0
    c0 = 1 / np.sqrt(eps0 * mu0)
    hbar = 1.0

    # TLS parameters
    d0 = 1e-2
    omega_TLS = 2 * np.pi * 1.0
    Gamma0 = (d0 ** 2) * (omega_TLS ** 3) / (3.0 * np.pi * hbar * eps0 * c0 ** 3)
    # We work in 3D, and dipole is in z direction

    # Time-step
    dx = 0.05
    dt = 0.56 * dx

    # Number of TLSs
    N_TLS = 1

    # Read in b data from file
    data = np.genfromtxt("data/b_FDTD_P0=1_dx=0.05.csv", dtype='complex', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_iter = data.shape[0]  # Total number of steps
    print(data.shape)
    
    t_arr = np.arange(1, max_iter + 1) * dt

    b_real = data[:, :].real
    b_imag = data[:, :].imag

    J = 2 * d0 * omega_TLS * b_imag / (dx ** 3)
    J = J.reshape(-1, 1)

    b = b_real + 1j * b_imag
    b_abs = np.sqrt(b_real ** 2 + b_imag ** 2)

    P_tot = np.sum(b_abs ** 2, axis=1)

    # Read in E field data from file
    data = np.genfromtxt("data/Ex.csv", dtype='float', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    Ex = data[:, :]
    
    # Visualize the time-evolution of b
    plt.figure(figsize=(10, 6))
    for id_TLS in range(N_TLS):
        plt.plot(t_arr, b_abs[:, id_TLS], label=str(id_TLS) + "-th TLS: Abs", linewidth=1)  # Plotting magnitude

        plt.plot(t_arr, b_real[:, id_TLS], label=str(id_TLS) + "-th TLS: Re", linewidth=1)  # Plotting real part

        plt.plot(t_arr, b_imag[:, id_TLS], label=str(id_TLS) + "-th TLS: Im", linewidth=1)  # Plotting imag part

    # plt.xlim([0, 10])
    # plt.ylim([0, np.max(b_abs)])

    plt.title("TLS: Time Evolution of b")
    plt.xlabel("Time (sec)")
    plt.ylabel("b")

    plt.legend()  # Adding a legend to differentiate the curves
    plt.grid(True)  # Adding grid for better readability
    plt.tight_layout()  # Automatically adjusts subplot params to give specified padding
    
    # Visualize the decay of total excitation number
    plt.figure(figsize=(2.5, 1.8))  # Enhanced figure size

    plt.plot(t_arr, P_tot, color='red', linewidth=3)  # Plotting magnitude
    # plt.plot(t_arr, np.exp(-Gamma0 * t_arr), label='exp[-Gamma * t]', linewidth=1)

    # Hide ticks and labels on both axes
    plt.xticks([])
    plt.yticks([])

    # Remove the borders of the figure
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    plt.xlim([0, 300])
    plt.ylim([0, 1.3])

    # plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    plt.show()


def TLS_PEC_dynamics_analyze():
    # Constants
    eps0 = 1.0
    mu0 = 1.0
    c0 = 1 / np.sqrt(eps0 * mu0)
    hbar = 1.0

    # TLS parameters
    d0 = 1e-2
    omega_TLS = 2 * np.pi * 1.0
    Gamma0 = (d0 ** 2) * (omega_TLS ** 3) / (3.0 * np.pi * hbar * eps0 * c0 ** 3)
    # We work in 3D, and dipole is in z direction

    # Time-step
    dx = 0.025
    dt = 0.56 * dx

    # Distance to PEC mirror
    z0 = 0.1

    # Define the integrand function
    # Based on eq.74 in Barnes' LDOS paper!
    def integrand(theta_k):
        term1 = np.sin(theta_k) - 0.5 * (np.sin(theta_k)**3)
        term2 = np.sin((omega_TLS / c0) * np.cos(theta_k) * z0)**2
        return term1 * term2

    # Perform the integral from 0 to pi/2
    result, error = quad(integrand, 0, np.pi/2)
    print('result:', result)
    Gamma_PEC = result * 3 * Gamma0  # The modified decay rate 
    
    # Read in b data from file
    data = np.genfromtxt("data/b_PEC_d=0.1.csv", dtype='complex', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_iter = data.shape[0]  # Total number of steps
    print(data.shape)
    
    t_arr = np.arange(1, max_iter + 1) * dt

    b_real = data[:, :].real
    b_imag = data[:, :].imag

    b = b_real + 1j * b_imag
    b_abs = np.sqrt(b_real ** 2 + b_imag ** 2)

    P_tot = np.sum(b_abs ** 2, axis=1)

    # Visualize the decay of total excitation number
    plt.figure(figsize=(10, 6))  # Enhanced figure size

    plt.plot(t_arr, P_tot, label='Total P', color='red', linewidth=1)  # Plotting magnitude
    plt.plot(t_arr, np.exp(-Gamma_PEC * t_arr), label='PEC', linewidth=1)
    plt.plot(t_arr, np.exp(-Gamma0 * t_arr), label='exp[-Gamma * t]', linewidth=1)

    # plt.xlim([0, 10])
    # plt.ylim([-1, 1])

    plt.title("TLS: decay of total excited probability")
    plt.xlabel("Time (sec)")
    plt.ylabel("log(P)")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
  

def Schrodinger_dynamics_analyze():
    # Constants
    eps0 = 1.0
    mu0 = 1.0
    c0 = 1 / np.sqrt(eps0 * mu0)
    hbar = 1.0

    # TLS parameters
    d0 = 2e-2
    omega_TLS = 2 * np.pi * 1.0
    Gamma0 = (d0 ** 2) * (omega_TLS ** 3) / (3.0 * np.pi * hbar * eps0 * c0 ** 3)
    # We work in 3D, and dipole is in z direction

    # Time-step
    dx = 0.05
    dt = 0.56 * dx

    # Read in b data from file
    data = np.genfromtxt("data/Pe_Schrodinger_P0=1_dx=0.05.csv", dtype='float', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_iter = data.shape[0]  # Total number of steps
    print(data.shape)
    
    t_arr = np.arange(1, max_iter + 1) * dt

    Pe = data[:, :]

    # Visualize the decay of total excitation number
    plt.figure(figsize=(2.5, 1.8))  # Enhanced figure size

    plt.plot(t_arr, Pe, color='red', linewidth=3)  # Plotting magnitude
    # plt.plot(t_arr, np.exp(-Gamma0 * t_arr), label='exp[-Gamma * t]', linewidth=1)

    # Hide ticks and labels on both axes
    plt.xticks([])
    plt.yticks([])

    # Remove the borders of the figure
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    plt.xlim([0, 300])
    plt.ylim([0, 1.3])

    # plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    plt.show()

def Bloch_dynamics_analyze():
    # Constants
    eps0 = 1.0
    mu0 = 1.0
    c0 = 1 / np.sqrt(eps0 * mu0)
    hbar = 1.0

    # TLS parameters
    d0 = 2e-2
    omega_TLS = 2 * np.pi * 1.0
    Gamma0 = (d0 ** 2) * (omega_TLS ** 3) / (3.0 * np.pi * hbar * eps0 * c0 ** 3)
    # We work in 3D, and dipole is in z direction

    # Time-step
    dx = 0.05
    dt = 0.56 * dx

    # Read in b data from file
    data = np.genfromtxt("data/Pe_Bloch_P0=1_dx=0.05.csv", dtype='float', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_iter = data.shape[0]  # Total number of steps
    print(data.shape)
    
    t_arr = np.arange(1, max_iter + 1) * dt

    Pe = data[:, :]

    # Visualize the decay of total excitation number
    plt.figure(figsize=(2.5, 1.8))  # Enhanced figure size

    plt.plot(t_arr, Pe, color='red', linewidth=3)  # Plotting magnitude
    # plt.plot(t_arr, np.exp(-Gamma0 * t_arr), label='exp[-Gamma * t]', linewidth=1)

    # Hide ticks and labels on both axes
    plt.xticks([])
    plt.yticks([])

    # Remove the borders of the figure
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    plt.xlim([0, 300])
    plt.ylim([0, 1.3])

    # plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    plt.show()


if __name__ == "__main__":
    # Plot the time-evolution of TLS
    TLS_dynamics_analyze()
    # TLS_PEC_dynamics_analyze()

    # Schrodinger_dynamics_analyze()

    # Bloch_dynamics_analyze()

