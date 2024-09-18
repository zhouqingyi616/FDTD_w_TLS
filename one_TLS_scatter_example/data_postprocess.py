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
    d0 = 2e-2
    omega_TLS = 2 * np.pi * 1.0
    Gamma0 = (d0 ** 2) * (omega_TLS ** 3) / (3.0 * np.pi * hbar * eps0 * c0 ** 3)
    # We work in 3D, and dipole is in z direction

    # Time-step
    dx = 0.05
    dt = 0.56 * dx

    # Number of TLSs
    N_TLS = 1

    # Read in b data from file
    data = np.genfromtxt("data/Pe.csv", dtype='complex', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_iter = data.shape[0]  # Total number of steps
    print(data.shape)
    
    t_arr = np.arange(1, max_iter + 1) * dt
    '''
    b = data[:, :]
    b_abs = np.sqrt(b.real ** 2 + b.imag ** 2)
    P_tot = np.sum(b_abs ** 2, axis=1)
    '''
    P_tot = data[:, :]

    # Read in E field data from file
    data = np.genfromtxt("data/E_drive.csv", dtype='float', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    Ex = data[:, :]
    
    # Visualize the time-evolution of b
    plt.figure(figsize=(6, 4))
    plt.plot(t_arr, Ex, label="E", linewidth=1)  # Plotting E field

    # plt.xlim([0, 10])
    # plt.ylim([0, np.max(b_abs)])

    plt.title("Driving term")
    plt.xlabel("Time (sec)")
    plt.ylabel("Ex")

    plt.legend()  # Adding a legend to differentiate the curves
    # plt.grid(True)  # Adding grid for better readability
    plt.tight_layout()  # Automatically adjusts subplot params to give specified padding
    
    # Visualize the decay of total excitation number
    plt.figure(figsize=(4, 2))  # Enhanced figure size

    plt.plot(t_arr, P_tot, color='red', linewidth=3)  # Plotting magnitude
    # plt.plot(t_arr, np.exp(-Gamma0 * t_arr), label='exp[-Gamma * t]', linewidth=1)

    # Hide ticks and labels on both axes
    plt.xticks([])
    # plt.yticks([])

    # Remove the borders of the figure
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    # plt.xlim([0, 300])
    plt.ylim([0, 1.2 * np.max(P_tot)])

    # plt.legend()
    plt.grid(False)
    plt.tight_layout()
    
    plt.show()


def cross_section_analyze():
    # Parameters of FDTD simulation
    dx = 0.05
    c0 = 1.0
    dt = 0.56 * dx / c0

    # Frequency range
    freq0 = 1.0
    freq_sigma = 0.05 * freq0
    t_sigma = 1 / freq_sigma / (2 * np.pi)
    t_peak = 5 * t_sigma
    Amp_inc_wave = 0.01

    # TLS parameters
    d0 = 2e-2
    omega_TLS = 2 * np.pi * freq0
    Gamma0 = (d0 ** 2) * (omega_TLS ** 3) / (3.0 * np.pi)
    lamda0 = c0 / freq0

    # The number of rows and columns
    max_iter = 53572
    max_iter_half = max_iter // 2
    save_len = 726

    # Read in the recorded E1, E2 field
    E1_file_name = 'data/E1_monitor.bin'
    E1_monitor_half = np.fromfile(E1_file_name, dtype=np.float32).reshape((max_iter_half, save_len))
    E2_file_name = 'data/E2_monitor.bin'
    E2_monitor_half = np.fromfile(E2_file_name, dtype=np.float32).reshape((max_iter_half, save_len))

    # Read in the recorded H1, H2 field
    H1_file_name = 'data/H1_monitor.bin'
    H1_monitor_half = np.fromfile(H1_file_name, dtype=np.float32).reshape((max_iter_half, save_len))
    H2_file_name = 'data/H2_monitor.bin'
    H2_monitor_half = np.fromfile(H2_file_name, dtype=np.float32).reshape((max_iter_half, save_len))

    # Use longer sequence to get higher frequency resolution
    max_iter = 5 * max_iter
    # Construct complete time sequence based on down-sampled data
    E1_monitor = np.zeros((max_iter, save_len))
    E2_monitor = np.zeros((max_iter, save_len))
    H1_monitor = np.zeros((max_iter, save_len))
    H2_monitor = np.zeros((max_iter, save_len))

    print("Reading FDTD data, interpolation...")
    # Fill in the entries of the new array
    for i in range(max_iter_half):
        # Copy data
        E1_monitor[2*i] = E1_monitor_half[i]
        E2_monitor[2*i] = E2_monitor_half[i]
        H1_monitor[2*i] = H1_monitor_half[i]
        H2_monitor[2*i] = H2_monitor_half[i]
        if i < max_iter_half - 1:
            # Average the i-th and (i+1)-th rows
            E1_monitor[2*i + 1] = (E1_monitor_half[i] + E1_monitor_half[i+1]) / 2
            E2_monitor[2*i + 1] = (E2_monitor_half[i] + E2_monitor_half[i+1]) / 2
            H1_monitor[2*i + 1] = (H1_monitor_half[i] + H1_monitor_half[i+1]) / 2
            H2_monitor[2*i + 1] = (H2_monitor_half[i] + H2_monitor_half[i+1]) / 2
    
    t_arr = np.arange(0, max_iter) * dt
    df = 1.0 / (max_iter * dt)

    print("Start calculating FFT...")
    # Now calculate the Fourier transform
    E1_monitor_FFT = np.fft.fft(E1_monitor, axis=0)
    E1_monitor_FFT = E1_monitor_FFT[:int((max_iter + 1) / 2), :]
    E1_monitor_FFT[1:, :] = 2 * E1_monitor_FFT[1:, :]

    E2_monitor_FFT = np.fft.fft(E2_monitor, axis=0)
    E2_monitor_FFT = E2_monitor_FFT[:int((max_iter + 1) / 2), :]
    E2_monitor_FFT[1:, :] = 2 * E2_monitor_FFT[1:, :]

    H1_monitor_FFT = np.fft.fft(H1_monitor, axis=0)
    H1_monitor_FFT = H1_monitor_FFT[:int((max_iter + 1) / 2), :]
    H1_monitor_FFT[1:, :] = 2 * H1_monitor_FFT[1:, :]

    H2_monitor_FFT = np.fft.fft(H2_monitor, axis=0)
    H2_monitor_FFT = H2_monitor_FFT[:int((max_iter + 1) / 2), :]
    H2_monitor_FFT[1:, :] = 2 * H2_monitor_FFT[1:, :]
    # Use the single-sided FFT result

    freq_num = E1_monitor_FFT.shape[0]  # Number of frequencies
    print('Number of frequencies:', freq_num)

    freq_arr = np.arange(0, freq_num) * df
    omega_arr = 2 * np.pi * freq_arr
    output_power_FFT = np.zeros(freq_num)
    
    for id_freq in range(freq_num):
        # Calculate output power
        poynting_tmp = np.sum(E1_monitor_FFT[id_freq, :] * np.conj(H1_monitor_FFT[id_freq, :]))
        poynting_tmp = poynting_tmp - np.sum(E2_monitor_FFT[id_freq, :] * np.conj(H2_monitor_FFT[id_freq, :]))
        poynting_tmp = poynting_tmp * dx * dx / 1
        # Average power = 1/2 * Re[E x H*]
        output_power_FFT[id_freq] = np.real(poynting_tmp) / 2

    # Now calculate the intensity of incident wave
    t_arr = np.arange(0, max_iter) * dt
    E_inc = Amp_inc_wave * np.exp(-(t_arr - t_peak) ** 2 / (2 * t_sigma ** 2)) * np.sin(2 * np.pi * freq0 * t_arr)

    # FFT
    E_inc_FFT = np.fft.fft(E_inc, axis=0)
    E_inc_FFT = E_inc_FFT[:int((max_iter + 1) / 2)]
    E_inc_FFT[1:] = 2 * E_inc_FFT[1:]

    inc_power_FFT = np.abs(E_inc_FFT) ** 2 / 2.0  # Since eta0 = 1.0
    
    # Plotting the input power & output power
    plt.figure(figsize=(10, 6))  # Sets the figure size similar to the previous plot
    plt.plot(omega_arr, output_power_FFT, label='output', color='red', linestyle='-')

    plt.plot(omega_arr, inc_power_FFT, label='input I0', color='black', linestyle='--')
    
    # plt.xlim([omega_TLS - 3 * Gamma0, omega_TLS + 3 * Gamma0])  # Adjust the x-axis limits
    # plt.ylim([0, sigma0])

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('P')
    plt.title('Power')
    plt.legend()

    # Scattering cross section
    sigma_FDTD = output_power_FFT / (inc_power_FFT + 1e-7)
    # Avoid instability

    sigma0 = 3 * lamda0 ** 2 / (2 * np.pi)

    # Plotting the scattering cross section
    plt.figure(figsize=(6, 4))  # Sets the figure size similar to the previous plot
    plt.plot(omega_arr, (sigma_FDTD / sigma0), color='red', linestyle='-', linewidth=3)
    
    # plt.xlim([omega_TLS - 3 * Gamma0, omega_TLS + 3 * Gamma0])  # Adjust the x-axis limits
    # plt.ylim([-4, 1])

    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('sigma')
    # plt.title('Cross section')
    # plt.legend()

    plt.axvline(x=omega_TLS, color='gray', linestyle='--', linewidth=4)  # Vertical line at omega_TLS
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=4)  # Horizontal line at y = 0

    # plt.xticks([])
    # plt.yticks([])

    # Remove the borders of the figure
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

    # Save the arrays to .npy files for future use
    np.save('data/omega_arr.npy', omega_arr)  # Saves omega_arr to a file named 'omega_arr.npy'
    np.save('data/sigma_FDTD.npy', sigma_FDTD)  # Saves sigma_FDTD to a file named 'sigma_FDTD.npy'


def plot_figures():
    # Constants
    eps0 = 1.0
    mu0 = 1.0
    c0 = 1 / np.sqrt(eps0 * mu0)
    hbar = 1.0

    # TLS parameters
    d0 = 2e-2
    omega_TLS = 2 * np.pi * 1.0
    Gamma0 = (d0 ** 2) * (omega_TLS ** 3) / (3.0 * np.pi * hbar * eps0 * c0 ** 3)
    lamda0 = c0 / (omega_TLS / (2 * np.pi))
    # We work in 3D, and dipole is in z direction

    # Time-step
    dx = 0.05
    dt = 0.56 * dx

    # Specify folder
    folder = "include_rad_zero_decay"
    # Read in Pe data from file
    data = np.genfromtxt("data/" + folder + "/Pe.csv", dtype='complex', delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    max_iter = data.shape[0]  # Total number of steps
    print(data.shape)
    
    t_arr = np.arange(1, max_iter + 1) * dt
    P_tot = data[:, :]

    # Visualize the decay of total excitation number
    plt.figure(figsize=(2.5, 1.8))  # Enhanced figure size

    plt.plot(t_arr, P_tot, color='red', linewidth=2)  # Plotting magnitude
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

    plt.xlim([0, 150])
    plt.ylim([0, 6.5e-7])

    plt.grid(False)
    plt.tight_layout()

    # Visualize the scattering cross section
    sigma0 = 3 * lamda0 ** 2 / (2 * np.pi)

    omega_arr = np.load("data/" + folder + "/omega_arr.npy")
    sigma_FDTD = np.load("data/" + folder + "/sigma_FDTD.npy")

    # Plotting the scattering cross section
    plt.figure(figsize=(3, 1.8))  # Sets the figure size similar to the previous plot
    plt.plot(omega_arr, 400 * (sigma_FDTD / sigma0), color='red', linestyle='-', linewidth=1.5)
    # 400
    # plt.xlim([omega_TLS - 3 * Gamma0, omega_TLS + 3 * Gamma0])
    plt.xlim([5, 7.5])
    plt.ylim([0, 1.03])

    plt.axvline(x=omega_TLS, color='gray', linestyle='--', linewidth=2)  # Vertical line at omega_TLS
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=2)  # Horizontal line at y = 0

    plt.xticks([])
    plt.yticks([])

    # Remove the borders of the figure
    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()


if __name__ == "__main__":
    # Plot the time-evolution of TLS
    # TLS_dynamics_analyze()

    # Calculate the scattering cross section
    # cross_section_analyze() 

    # Visualization
    plot_figures()

