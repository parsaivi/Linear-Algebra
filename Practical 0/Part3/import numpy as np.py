import numpy as np

def convolve_1d_one_loops(signal, kernel):
    """
    Convolve a 1D signal with a kernel using a single loop (optimized version).

    Args:
        signal: 1D array representing the input signal.
        kernel: 1D array representing the filter to apply to the signal.

    Returns:
        output: 1D array representing the convolution result.
    """
    signal_len = len(signal)
    kernel_len = len(kernel)
    pad_size = kernel_len - 1

    # Step 1: Zero-pad the signal on both sides
    padded_signal = np.zeros(signal_len + 2 * pad_size)
    padded_signal[pad_size:pad_size + signal_len] = signal
    
    # Step 2: Reverse the kernel for convolution
    reversed_kernel = kernel[::-1]

    output_len = signal_len + kernel_len - 1
    output = np.zeros(output_len)

    # Step 3: Perform convolution using a single loop and vectorized operations like sum
    for i in range(output_len):
        output[i] = np.sum(padded_signal[i:i + kernel_len] * reversed_kernel)
    return output

# مثال
signal = np.array([1, 2, 3, 4, 5])
kernel = np.array([1, 0, -1])

result = convolve_1d_one_loops(signal, kernel)
print("Signal:", signal)
print("Kernel:", kernel)
print("Result:", result)