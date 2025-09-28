# verify_fft.py
import numpy as np

N = 524288  # must match the C++ program

# triple-sine input
n = np.arange(N, dtype=np.float64)
x = np.sin((100.0 * np.pi / N) * n) \
  + np.sin((1000.0 * np.pi / N) * n) \
  + np.sin((2000.0 * np.pi / N) * n)

# reference FFT (numpy uses natural-order output)
ref = np.fft.fft(x.astype(np.float64))

# load C++ results (real,imag per line)
cpp = np.loadtxt("fft_out.csv", delimiter=",", dtype=np.float64)
cpp_c = cpp[:,0] + 1j*cpp[:,1]

# compare
abs_err = np.abs(cpp_c - ref)
max_err = abs_err.max()
rms_err = np.sqrt((abs_err**2).mean())

print(f"N={N}  max_abs_err={max_err:.3e}  rms_err={rms_err:.3e}")

