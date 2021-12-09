import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
def Fft_series(series):
    """
      The `FFT y[k]` of length N of the length-N sequence `x[n]` is defined as

        $$ y[k]= \sum_{n=0}^{N-1} e^{-2\pi i \frac{kn}{N} } x[n] $$
    """
    
    N = len(series)
    y_r = np.zeros(len(series))
    y_i = np.zeros(len(series))
    
    for k in range(N):
        y_r[k] = sum([(np.exp(-2*pi*1j*k*n/N)*series[n]).real for n in range(N)])
        y_i[k] = sum([(np.exp(-2*pi*1j*k*n/N)*series[n]).imag for n in range(N)])
    return y_r + 1j*y_i
                   
    
def inverse_Fft_series(fft_series): 
  """
      The inverse transform of the series

        $$ x[n]= \dfrac{1}{N} \sum_{k=0}^{N-1} e^{2\pi i \frac{kn}{N} } y[k] $$
  """
  
    N = len(fft_series)
    y_r = np.zeros(len(fft_series))
    y_i = np.zeros(len(fft_series))
    
    for k in range(N):
        y_r[k] = sum([(np.exp(2*pi*1j*k*n/N)*fft_series[n]).real for n in range(N)])
        y_i[k] = sum([(np.exp(2*pi*1j*k*n/N)*fft_series[n]).imag for n in range(N)])
        
    return y_r + 1j*y_i
