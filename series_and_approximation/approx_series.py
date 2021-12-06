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
    ax.plot(x, fx(x), label = 'True_f', linewidth = 2)
    ax.grid(color = 'violet', linestyle='--', linewidth = 1)
    ax.plot(x, dif_f, label = 'deriv(f, deg={})'.format(n_deg), linewidth = 2)
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

# /=================================================================================================\
def sequence_n_epsilon(n_min, n_max, n_steps, your_seq, seq_limit = 0, img_loc = 'center right'):
    """
    
    """
    eps = []
    
    for n in range(n_min, n_max + n_steps // 2, n_steps):
        
        err = abs(your_seq(n) - seq_limit)
        eps.append(err)
        
    fig, ax1 = plt.subplots(1, 1, figsize = (20, 6), dpi = 90)
    color = 'tab:red'
    ax1.set_xlabel('number of n')
    ax1.set_ylabel('epsilon = |x_n - $\overline{x}$|', color=color)
    y = eps
    x = range(n_min, n_max + n_steps // 2, n_steps)
    ax1.plot(x, y, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title("your input series and the convergence_result")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel("relative_error", color=color)  # we already handled the x-label with ax1
    z = [your_seq(n) for n in x]
    zs = [seq_limit for n in x]
    ax2.plot(x, z, color=color, label = '$x_n$ values')
    ax2.plot(x, zs, color='black', label = 'sequence_limit')
    ax2.legend(loc = img_loc)
    ax2.tick_params(axis='y', labelcolor=color)    
    ax1.grid(color='violet', linestyle='--', linewidth=1)
    ax2.grid(color='violet', linestyle='--', linewidth=1)

# /=================================================================================================\
def series_n_epsilon(n_min, n_max, n_steps, your_series, series_limit = 0, img_loc = 'center right'):
    
    eps = []
    
    for n in range(n_min, n_max + n_steps // 2, n_steps):
        
        err = abs(your_series(n) - series_limit)
        eps.append(err)
        
    fig, ax1 = plt.subplots(1, 1, figsize = (20, 6), dpi = 90)
    color = 'tab:red'
    ax1.set_xlabel('number of n')
    ax1.set_ylabel('epsilon = |x_n - $\overline{x}$|', color=color)
    y = eps
    x = range(n_min, n_max + n_steps // 2, n_steps)
    ax1.plot(x, y, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title("your input series and the convergence_result")
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel("relative_error", color=color)  # we already handled the x-label with ax1
    z = [your_series(n) for n in x]
    zs = [series_limit for n in x]
    ax2.plot(x, z, color=color, label = '$x_n$ values')
    ax2.plot(x, zs, color='black', label = 'series_limit')
    ax2.legend(loc = img_loc)
    ax2.tick_params(axis='y', labelcolor=color)    
    ax1.grid(color='violet', linestyle='--', linewidth=1)
    ax2.grid(color='violet', linestyle='--', linewidth=1)

# /=================================================================================================\
def sigmoid(x): return 1 / (1 + np.exp(-x))

# /=================================================================================================\
def n_layers_fx(fx, x, n):
    
    y = fx(x)
    if n == 1:
        return fx(x)
    else:
        for k in range(n):
            y = fx(y)
    return y

# /=================================================================================================\
def depth_view_actv_func(fx, x, n_list, func_name):
    
    fig, ax = plt.subplots(1, 1, figsize = (20, 8), dpi = 100)
    for n in n_list:
        y = [n_layers_fx(fx, t, n = n) for t in x]
        ax.plot(x, y, label = 'n_layers={}'.format(n), linewidth = 2)
        ax.grid(color = 'cyan', linestyle='--', linewidth = 1)

    plt.title("$f_1(x) =$ {}".format(func_name) +"$(x)$\n $f_2(x) = f_1(f_1(x))$\n ...\n $f_n(x) = f_1(f_{n-1}(x))$\n where n is the number of layers")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
    
# /=================================================================================================\    
def integral_approx(fx, a, b, n):
    r = (b-a)/100
    x_ab = np.linspace(a, b, 201)
    x_t = np.linspace(a-r, b+r, 201)

    h = 1/(np.sqrt(n))
    u = np.linspace(a, b, n+1)
    u_s = u[1:]
    
    s = 0
    for k in range(len(u)-1):
        width = u[k+1] - u[k]
        height = (fx(u[k]) + fx(u[k+1]))/2
        s += width * height
    
    return s

# /=================================================================================================\
def loss_integral_Riemann_sum(fx, a, b, min_n, max_n, f_name):
    
    from scipy.integrate import quad   
    sn = []
    sx = range(min_n, max_n + 1)    
    sn = [integral_approx(fx, a, b, n) for n in sx]
    
     ## check odd or even function
    x_c = np.linspace(0, 10, 100) 
    if (fx(x_c) == fx(-x_c)).all(): 
        s0 = np.round(quad(fx, a, b)[0], 15)
    elif (fx(x_c) == -fx(-x_c)).all():
        s0 = 0
    else:
        s0 = np.round(quad(fx, a, b)[0], 12)
    
    plt.figure(figsize=(20, 3.5), dpi=90)
    plt.plot(sx, sn, 'o-', label = 'S(f, a, b, n)_approx')
    plt.plot([min_n, max_n], [s0, s0], '-', label = 'true_integral')
    plt.grid(color='violet', linestyle='--', linewidth=1)
    plt.xlabel("n_partitions")
    plt.title(f_name)
    plt.legend()
    
# /=================================================================================================\
def rect_kernel(t):
    return 0.5 * (abs(t) <= 1)

def integral_plot(fx, a, b, n, ax, S_clr = "lightsalmon", S_apr_clr = "#C79FEF"):
    r = (b-a)/100
    x_ab = np.linspace(a, b, 201)
    x_t = np.linspace(a-r, b+r, 201)

    h = 1/(np.sqrt(n))
    u = np.linspace(a, b, n+1)
    u_s = u[1:]
    
    s = 0
    for k in range(len(u)-1):
        ck = (fx(u[k]) + fx(u[k+1]))/2
        ax.fill_between([u[k], u[k+1]], [ck, ck], [fx(u[k]), fx(u[k+1])],
                         facecolor = S_clr, interpolate=True)
        ax.fill_between([u[k], u[k+1]], 
                      0,
                      [ck, ck],
                     facecolor = S_apr_clr, step = 'pre', edgecolor='b'
                    )
    ax.plot([a, b], [0, 0], color = 'green', linewidth = 2.5, label = '$S_n(f,a,b)$_approx')
    ax.plot(x_t, fx(x_t), '--', color = 'darkgreen', linewidth = 0.5, label = 'f(x)')
    ax.plot(x_ab, fx(x_ab), color = 'green', linewidth = 3)
    ax.plot([b, b], [0, fx(b)], color = 'green', linewidth = 3)
    ax.plot([a, a], [0, fx(a)], color = 'green', linewidth = 3)
    ax.grid(color='cyan', linestyle='--', linewidth=0.5)
    ax.set_title("n={}".format(n))
    ax.legend()
    
# /=================================================================================================\   
def koch_snowflake(order, scale=10):
    """
    Return two lists x, y of point coordinates of the Koch snowflake.

    Parameters
    ----------
    order : int
        The recursion depth.
    scale : float
        The extent of the snowflake (edge length of the base triangle).
    """
    def _koch_snowflake_complex(order):
        if order == 0:
            # initial triangle
            angles = np.array([0, 120, 240]) + 90
            return scale / np.sqrt(3) * np.exp(np.deg2rad(angles) * 1j)
        else:
            ZR = 0.5 - 0.5j * np.sqrt(3) / 3

            p1 = _koch_snowflake_complex(order - 1)  # start points
            p2 = np.roll(p1, shift=-1)  # end points
            dp = p2 - p1  # connection vectors

            new_points = np.empty(len(p1) * 4, dtype=np.complex128)
            new_points[::4] = p1
            new_points[1::4] = p1 + dp / 3
            new_points[2::4] = p1 + dp * ZR
            new_points[3::4] = p1 + dp / 3 * 2
            return new_points

    points = _koch_snowflake_complex(order)
    x, y = points.real, points.imag
    return x, y
