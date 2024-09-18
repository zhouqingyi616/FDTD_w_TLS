import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def system_transmission_calc(get_background):
    # Parameters of FDTD simulation
    dx = 0.04
    c0 = 1.0
    dt = 0.56 * dx / c0

    # Frequency range
    freq0 = c0 / 1.55
    freq_sigma = 0.1 * freq0

    # The number of rows and columns
    max_iter = 800000
    save_len = 216

    # Read in the recorded Ey, Ez field
    Ey_file_name = 'data/Ey_output_monitor.bin'
    Ey_output_monitor = np.fromfile(Ey_file_name, dtype=np.float32).reshape((max_iter, save_len))
    Ez_file_name = 'data/Ez_output_monitor.bin'
    Ez_output_monitor = np.fromfile(Ez_file_name, dtype=np.float32).reshape((max_iter, save_len))

    # Read in the recorded Hy, Hz field
    Hy_file_name = 'data/Hy_output_monitor.bin'
    Hy_output_monitor = np.fromfile(Hy_file_name, dtype=np.float32).reshape((max_iter, save_len))
    Hz_file_name = 'data/Hz_output_monitor.bin'
    Hz_output_monitor = np.fromfile(Hz_file_name, dtype=np.float32).reshape((max_iter, save_len))

    print("Number of iterations:", max_iter)
    print("Number of grid points inside monitor:", save_len)

    t_arr = np.arange(0, max_iter) * dt
    df = 1.0 / (max_iter * dt)

    # Now calculate the Fourier transform
    Ey_output_FFT = np.fft.fft(Ey_output_monitor, axis=0)
    Ey_output_FFT = Ey_output_FFT[:int((max_iter + 1) / 2), :]
    Ey_output_FFT[1:, :] = 2 * Ey_output_FFT[1:, :]

    Ez_output_FFT = np.fft.fft(Ez_output_monitor, axis=0)
    Ez_output_FFT = Ez_output_FFT[:int((max_iter + 1) / 2), :]
    Ez_output_FFT[1:, :] = 2 * Ez_output_FFT[1:, :]
    
    Hy_output_FFT = np.fft.fft(Hy_output_monitor, axis=0)
    Hy_output_FFT = Hy_output_FFT[:int((max_iter + 1) / 2), :]
    Hy_output_FFT[1:, :] = 2 * Hy_output_FFT[1:, :]

    Hz_output_FFT = np.fft.fft(Hz_output_monitor, axis=0)
    Hz_output_FFT = Hz_output_FFT[:int((max_iter + 1) / 2), :]
    Hz_output_FFT[1:, :] = 2 * Hz_output_FFT[1:, :]
    # Use the single-sided FFT result

    freq_num = Ey_output_FFT.shape[0]  # Number of frequencies

    freq_arr = np.arange(0, freq_num) * df
    output_power_FFT = np.zeros(freq_num)

    for id_freq in range(freq_num):
        # Calculate output power
        poynting_tmp = np.sum(Ey_output_FFT[id_freq, :] * np.conj(Hz_output_FFT[id_freq, :]))
        poynting_tmp = poynting_tmp - np.sum(Ez_output_FFT[id_freq, :] * np.conj(Hy_output_FFT[id_freq, :]))
        poynting_tmp = poynting_tmp * dx / max_iter
        # Average power = 1/2 * Re[E x H*]
        output_power_FFT[id_freq] = np.real(poynting_tmp) / 2

    if get_background:
        # Save data for power normalization
        power_normalize = output_power_FFT
        np.save('data/power_normalize.npy', power_normalize)

        # Visualize power spectrum
        plt.figure(figsize=(10, 6))  # Optional: Sets the figure size
        plt.plot(freq_arr, power_normalize, label='Output Power FFT', color='blue', linestyle='-')
        plt.xlim([freq0 - 3 * freq_sigma, freq0 + 3 * freq_sigma])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Output Power')
        plt.title('Waveguide: power spectrum')
        plt.legend()
        # plt.grid(True)
        plt.show()
    else:
        # Save the power spectrum
        np.save('data/power_N=12.npy', output_power_FFT)
        # Calculate the transmission spectrum
        power_normalize = np.load('data/power_normalize.npy')
        transmission = np.divide(output_power_FFT, power_normalize + 0)

        # Plotting both output power & normalize power
        plt.figure(figsize=(10, 6))  # Sets the figure size similar to the previous plot
        plt.plot(freq_arr, output_power_FFT, label='Output Power FFT', color='red',
                 linestyle='-')  # Output power (in red)
        plt.plot(freq_arr, power_normalize, label='Power Normalize', color='black',
                 linestyle='--')  # Power normalize (in black dashed line)

        plt.xlim([freq0 - 2 * freq_sigma, freq0 + 2 * freq_sigma])  # Adjust the x-axis limits

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Output power: comparison')
        plt.legend()
        # plt.grid(True)  # Uncomment if you prefer to have the grid

        # Visualize transmission spectrum
        plt.figure(figsize=(10, 6))  # Optional: Sets the figure size
        plt.plot(freq_arr, transmission, label='Transmission', color='blue', linestyle='-')
        plt.xlim([freq0 - 0.5 * freq_sigma, freq0 + 0.5 * freq_sigma])
        plt.ylim([0, 1.05])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('T')
        plt.title('Waveguide + ring resonator: transmission')
        plt.legend()
        # plt.grid(True)

        plt.show()


def visualize_Rabi_splitting():
    N_TLS_arr = np.array([1, 2, 3, 4, 6, 9, 16])
    Rabi_splitting_arr = 0.001 * np.array([1, 1.4, 1.7, 2.1, 2.6, 3.1, 4.15])

    # Define the model function y = k * x^p
    def model_func(x, k, p):
        return k * x**p

    # Curve fitting
    params, params_covariance = curve_fit(model_func, N_TLS_arr, Rabi_splitting_arr)

    # Generate data for the fitted curve
    x_fit = np.linspace(N_TLS_arr.min(), N_TLS_arr.max(), 100)
    y_fit = model_func(x_fit, *params)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(N_TLS_arr, Rabi_splitting_arr, color='red', label='Data Points')
    # Plot of the fitted curve
    plt.plot(x_fit, y_fit, label='Fitted Curve $y = k \cdot x^p$', color='blue')

    # Setting log scale for both axes
    # plt.xscale('log')
    # plt.yscale('log')

    # Labels and legend
    plt.xlabel('N_TLS')
    plt.ylabel('Rabi Splitting')
    plt.title('Log-Log Plot with Fitted Curve')
    plt.legend()

    plt.grid(True, which="both", ls="--")

    print('power law:', params[1])
    plt.show()
    

if __name__ == "__main__":
    # Calculate the transmission
    get_background = False  # True: bare waveguide; False: w/ ring & TLS
    system_transmission_calc(get_background=get_background)

    # Verify the sqrt[N] relationship of Rabi splitting
    # visualize_Rabi_splitting()

