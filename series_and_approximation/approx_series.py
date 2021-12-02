"""
    This package is used to generate the values of the
    
      1) derivative of function
      2) Taylor series
      3) Fourier series
      4) Legendre approximation
      5) Larange approximation
      6) Series converenge viewing
      7) Limit of a given sequence
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.interpolate import approximate_taylor_polynomial
sns.set_theme(style = "darkgrid", 
                  rc = {"axes.spines.right": False, "axes.spines.top": False})

# /=================================================================================================\    
def derivative_view(fx, x, n_deg, ax):
    """
    """
    order = 1 + 2*n_deg
    dif_f = [round(derivative(fx, x0=t, n=n_deg, dx=5e-2, order=order), 8) for t in x]
    ax.plot(x, fx(x), label = 'True_f')
    ax.grid(color='r', linestyle='--', linewidth=1)
    ax.plot(x, dif_f, label = 'deriv(f, deg={})'.format(n_deg) )
    ax.set_xlabel("x")
    ax.set_ylabel("f and its derivative")
    ax.set_title("degree = {}".format(n_deg) )
    ax.legend()

# /=================================================================================================\    
def Fourier_series_approx(x_range, fx, n_deg):
    """
        This function is used to create the Fourier series which be truncated
        
            fourier_f(x) = sum(2*cn*exp(j*n*x/L))
        where
            cn is the Fourier coefficience
    """
    y = fx(x_range)
    period = 2*np.pi
    time = x_range
    
    def cn(n, y):        
        c = y*np.exp(-1j*2*n*np.pi*time/ period)
        return c.sum()/c.size

    def f(x_range, Nh):
        f = np.array([2*cn(i, y)*np.exp(1j*2*i*np.pi*x_range / period) for i in range(1,Nh+1)])
        return f.sum()    
    
    fft_series = np.array([f(t, n_deg).real for t in x_range])
    
    return fft_series

# /=================================================================================================\    
def Fourier_series_approx_view(x, fx, n_deg, ax):


    y = fx(x)
    y2 = Fourier_series_approx(x, fx, n_deg)
    ax.grid(color='violet', linestyle='--', linewidth=1)
    ax.plot(x, y, label = 'True_f', color = 'darkred', linewidth = 2)
    ax.plot(x, y2, label = 'truncate_{}'.format(n_deg), color = 'darkblue', linewidth = 2)
    ax.legend()
    
# /=================================================================================================\    
def Fourier_series_approx_loss(x_range, fx, min_deg, max_deg):

    y = fx(x_range)
    rel_errs = []
    mse_errs = []
    
    for k in range(min_deg, max_deg + 1):
        approx_values = Fourier_series_approx(x_range, fx, n_deg = k)
        
        mse = np.mean((approx_values - y)**2)
        mse_errs.append(mse)
        
        denom = np.array([1e-1 if x == 0 else x for x in y])
        mpe = np.mean(np.abs( (approx_values - y) / denom))
        rel_errs.append(mpe)

    fig, ax1 = plt.subplots(1,1, figsize=(20,5), dpi=100)
    color = 'tab:red'
    ax1.set_xlabel('number of truncated_degrees')
    ax1.set_ylabel('MSE', color=color)
    ax1.plot(range(min_deg, max_deg + 1), rel_errs, 'o-', color='darkred', label='relative_errors', linewidth = 2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title("MSE & relative_error of Fourier_approximation w.r.t degree of truncated")
    ax1.grid(color='violet', linestyle='-.', linewidth=1)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel("relative_error", color=color)  # we already handled the x-label with ax1
    ax2.plot(range(min_deg, max_deg + 1), mse_errs, 'o-', color='darkgreen', label='mse', linewidth = 2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(color='violet', linestyle='-.', linewidth=1)
    
# /=================================================================================================\    
def taylor_series_approx(x_range, fx, n_deg):
    """
    """
    
    real_val = fx(x_range)
    taylor_approx = approximate_taylor_polynomial(fx, 0, degree = n_deg, scale=1, order = n_deg+5)
    approx_values = taylor_approx(x_range)
    
    return approx_values

# /=================================================================================================\    
def taylor_approx_view(x_range, fx, n_deg, ax):
    """
    """
    
    y = fx(x_range)
    y2 = taylor_series_approx(x_range, fx, n_deg)
    ax.grid(color='violet', linestyle='--', linewidth=1)
    ax.plot(x_range, y, label = 'True_f', color = 'darkred', linewidth = 2)
    ax.plot(x_range, y2, label = 'Taylor_(f, deg={})'.format(n_deg), color = 'darkblue', linewidth = 2)
    ax.legend()
    
# /=================================================================================================\    
def taylor_loss_view(x_range, fx, min_deg, max_deg):
    """
    
    """
    
    real_val = fx(x_range)
    mse_vals = []
    rel_vals = []
    p = 1 + (max_deg - min_deg) // 5
    fig, ax1 = plt.subplots(1, 1, figsize = (20, 6), dpi = 90)
    for k in range(min_deg, max_deg + 1):
        
        approx_values = taylor_series_approx(x_range, fx, n_deg = k)
        mse = np.mean((approx_values - real_val)**2)
        mse_vals.append(mse)
        
        denom = np.array([1e-1 if x == 0 else x for x in real_val])
        mpe = np.mean(np.abs( (approx_values - real_val) / denom))
        rel_vals.append(mpe)
                
    color = 'tab:red'
    ax1.set_xlabel('number of degrees')
    ax1.set_ylabel('MSE', color=color)
    y = mse_vals
    x = range(min_deg, max_deg + 1)
    ax1.plot(x, y, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title("MSE & relative_error of Taylor_approximation w.r.t degree of polynomial")
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel("relative_error", color=color)  # we already handled the x-label with ax1
    z = rel_vals
    ax2.plot(x, z, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.grid(color='violet', linestyle='--', linewidth=1)
    ax2.grid(color='violet', linestyle='--', linewidth=1)
