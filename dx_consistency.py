import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time as clock
import multiprocessing as mp
from itertools import product
from scipy import fft
from scipy.stats import linregress as lin_fit
import the_functions as funcs

k = 0.3     #wavenumber
ac = -1     #cubic nonlinearity parameter

#NLS parameters:
w = np.sqrt(k**2 + 1)    #frequency 
c = k/w                  #group velocity

v1, v2, v3 = 2*w, 1-c**2 ,  3
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 


Lx = 1000       #boundaries for the x-space 
dt = 0.01
t0, T = 0., 1   #initial standard time and slow final time


e_list = np.linspace(0.04, 0.1, 4)
e_list0 = np.append(0, e_list)

#returns the superior error for a given dx and e value
#title = "soliton"
title = "defocusing square"
def dx_consistency(vals):
    dx, e = vals
    
    #x array calculation
    Nx =int(2*Lx/dx)
    x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx
    dx = x[1]-x[0]
    
    #canstants reordering
    values = A, B, gamma, e, k, w, c, ac, v1, v2, v3
    values_env = values[:3]
    values_NLS = values[:7]
    """
    #select initial condition solution
    env  = funcs.soliton(e*x, values_env)
    u0 =    e*funcs.NLS_approx(x, values_NLS, funcs.soliton)
    v0 =    e*funcs.NLS_approx_dt(x, values, funcs.soliton)
    
    """
    env  = funcs.square(e*x, values_env)
    u0 =    e*funcs.NLS_approx(x, values_NLS, funcs.square)
    v0 =    e*funcs.NLS_approx_dt(x, values, funcs.square)
    
    #setting up for time evolution
    t, tf, u, v = 0., (T/e**2), u0, v0
    
    k1 = 2*np.pi*fft.fftfreq(Nx, d=dx)
    K1 = k1/e
    K2 = K1**2
    
    values_strang = [e, c, ac, v1, v2, v3, K1, K2]
     
    superior_error = 0
    superior_H_error = 0
     
    vals_rk4 = [dx, ac]
    
    #time evolution
    while t < tf:
    
        t, u, v = funcs.RK4(t, u, v, dt, vals_rk4, funcs.NLKG)
        
        env = funcs.Strang_splitting(env, dt, values_strang)
        
        u_approx = e*env*funcs.Exp(x, t, k, w)    
        u_approx += np.conj(u_approx)
        
        error = max(abs(u-u_approx))
        H_error = funcs.Hp_norm(u-u_approx, 1, x) #sobolev norm error
    
        if error> superior_error: superior_error = error 
        if H_error> superior_H_error: superior_H_error = H_error 
    
    print(f"done with {dx, e}")
    
    #superior errors in C^0 and H^1 spaces returned based on dx and e 
    return superior_error, superior_H_error

#returns the fitting parameters for a given list of superior values over epsilon
def fitter(sup_err_list):
    sup_err_list0 = np.append(0, sup_err_list)

    fitter = lambda x, C, b: C*x**b

    popt, pcov = curve_fit(fitter, e_list0, sup_err_list0)
    C , b = popt
    return C, b

if __name__ == '__main__':
    
    #dx_list, based on the number of desired points
    dx_n = 3
    dx_list = np.linspace(0.02, 0.05, dx_n)
    
    #e_list and dx_list are "merged" to give a list of all possible (dx, e) pairs. 
    couple = list(product(dx_list, e_list))
    
    i = clock()
    #parrallel calculation of the resulting superior errors:
    p = mp.Pool()
    result = p.map(dx_consistency, couple)
    p.close()
    p.join()
    
    #superior errors divided in H1 and C^0 norms results
    result = np.array(result)
    sup_err_res, sup_H_err_res = np.array(result).T
    
    #lists containing lists of superior errors, each for a different dx step
    sup_err_lists_set = np.split(sup_err_res, dx_n)
    sup_H_err_lists_set = np.split(sup_H_err_res, dx_n)
    
    f = clock()
    
    print(f"time taken = {f-i} s")
    
    #calculate fitting parameters for each set of superior errors associated to
    #a specific dx:
    p = mp.Pool()
    CB_parameters = p.map(fitter, sup_err_lists_set)
    p.close()
    p.join()

    C_list, b_list = np.array(CB_parameters).T
    
    #mean C and b fitting parameters results.
    Ch = np.mean(C_list).round(2)    
    bh = np.mean(b_list).round(2)
    
    #only intercept in value at dx = 0
    b0 = lin_fit(dx_list, b_list)[1]
    
    #plot superior error fitting parameters over imposed dx
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title("stability of fit parameters, "+title+" solution")
    ax1.plot(dx_list, C_list, "b.-", label = f"C = {Ch}")
    ax1.set_ylim(Ch - 0.1, Ch + 0.1)
    ax1.set(ylabel= "C")
    ax1.legend()
    ax2.plot(dx_list, b_list, "r.-", label = f"b = {bh}, b = {round(b0, 2)} for dx=0")
    ax2.set_ylim(bh - 0.1, bh + 0.1)
    ax2.set(xlabel = "dx", ylabel= "b")
    ax2.legend()
    
    #repeat procedure, but for H1 sobolev norm
    p = mp.Pool()
    CB_parameters = p.map(fitter, sup_H_err_lists_set)
    p.close()
    p.join()

    C_list, b_list = np.array(CB_parameters).T
    
    Ch = np.mean(C_list).round(2)    
    bh = np.mean(b_list).round(2)
    
    #only intercept in value at dx = 0
    ip = lin_fit(dx_list, b_list)[1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.set_title("stability of fit parameters, H1 norm, "+title+" solution")
    ax1.plot(dx_list, C_list, "b.-", label = "C = {}".format(Ch))
    ax1.set_ylim(Ch - 0.1, Ch + 0.1)
    ax1.set(ylabel= "C")
    ax1.legend()
    ax2.plot(dx_list, b_list, "r.-", label = "b = {}, b = {} for dx=0".format(bh, round(ip, 2)))
    ax2.set_ylim(bh - 0.1, bh + 0.1)
    ax2.set(xlabel = "dx", ylabel= "b")
    ax2.legend()
