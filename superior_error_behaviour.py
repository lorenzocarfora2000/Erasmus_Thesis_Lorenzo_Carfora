import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time as clock
import multiprocessing as mp
import the_functions as funcs
from scipy import fft

k = 0.3   #Wavenumber
ac = -1   #Cubic nonlinearity sign

#NLS parameters:
w = np.sqrt(k**2 + 1)    #Frequency 
c = k/w                  #Group velocity

v1, v2, v3 = 2*w, 1-c**2 ,  3
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 


##Spatial array
Lx = 1000
dx = 0.02

Nx =int(2*Lx/dx)

x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx
dx = x[1]-x[0]

T = 1.        #Slow final time
dt = 0.015    #Standard time step

#Fourier k-space array
k1 = 2*np.pi*fft.fftfreq(Nx, d=dx)

vals_rk4 = [dx, ac]

#title = "soliton"
#title = "square"
title = "defocusing square"

#Calculates the superior error in C^0 and H^1 space of a given initial
#condition solution for a given epsilon value.
def sup_err_calc(e):
    
    #Reordering constants
    vals = A, B, gamma, e, k, w, c, ac, v1, v2, v3
    vals_env = vals[:3]
    vals_NLS = vals[:7]
    
    #Select which initial condition solution to model
    """
    env =   funcs.soliton(e*x, vals_env)
    u0  = e*funcs.NLS_approx(x, vals_NLS, funcs.soliton)
    v0  = e*funcs.NLS_approx_dt(x, vals, funcs.soliton)
    """

    env =   funcs.square(e*x, vals_env)
    u0  = e*funcs.NLS_approx(x, vals_NLS, funcs.square)
    v0  = e*funcs.NLS_approx_dt(x, vals, funcs.square)
    
    t, u, v, tf = 0., u0, v0, T/(e**2)
    
    #Fourier space associated to the large space X
    K1 = k1/e
    K2 = K1**2
    
    vals_strang = [e, c, ac, v1, v2, v3, K1, K2]
    
    superior_error   = 0
    superior_H_error = 0
    
    #Numerical evolution routine as shown in time_evolution.py
    while t < tf:
        
        #Numerical NLKGE
        t, u, v = funcs.RK4(t, u, v, dt, vals_rk4, funcs.NLKG)
        
        #Numerical NLSE for the envelope
        env = funcs.Strang_splitting(env, dt, vals_strang)
        
        #Approximation calculation based on the envelope function
        u_approx = e*env*funcs.Exp(x, t, k, w)  
        u_approx += np.conj(u_approx)
        
        #C^0 and H^1 norms of the approximation error
        error = max(abs(u-u_approx))
        H_error = funcs.Hp_norm(u-u_approx, 1, x)
    
        if error> superior_error: superior_error = error 
        if H_error> superior_H_error: superior_H_error = H_error 
        
    print(f"done with {e}")
    return superior_error, superior_H_error

#List of epsilon parameters
e_list = np.linspace(0.04, 0.1, 4)

if __name__ == '__main__':
    
    start = clock()
    
    #Parallel calculation, sup_err_set contains all the results for all the
    #e_list parameters
    p = mp.Pool()
    sup_err_set = p.map(sup_err_calc,  e_list)
    p.close()
    p.join()
    end = clock()
    
    #Separating results in lists for C^0 and H^1 norms respectively. 
    sup_err_list, sup_H_err_list = np.array(sup_err_set).T
    
    print(f"time taken: {end-start} seconds")
    
    #Append the trivial solution of e = 0.
    e_list = np.append(0, e_list)
    sup_err_list = np.append(0, sup_err_list)
    sup_H_err_list = np.append(0, sup_H_err_list)

    fitter = lambda x, a, b: a*x**b #Fitting function 
    
    #Fitting and plotting for the error in C^0
    popt, pcov = curve_fit(fitter, e_list, sup_err_list)
    x_fit = np.linspace(0, e_list[-1], 100)
    y_fit = fitter(x_fit, *popt)

    C, b = np.round(popt, 2)

    plt.figure(3)
    plt.plot(e_list, sup_err_list, ".", label="numerical")
    plt.xlabel("Amplitude parameter $\epsilon$, 0 < $\epsilon$ <<1")
    plt.ylabel("Superior error")
    plt.title(f"Error for cubic NLKG, $T_0$ ={T}, "+title) 
    plt.plot(x_fit, y_fit, "--", label = "fit = $C\epsilon^b$, with C = {}, b = {}".format(C, b) )
    plt.legend(loc = "best")
    
    #Fitting and plotting for the error in H^1
    poptH, pcovH = curve_fit(fitter, e_list, sup_H_err_list)
    y_fit = fitter(x_fit, *poptH)

    CH, bH = np.round(poptH, 2)

    plt.figure(4)
    plt.plot(e_list, sup_H_err_list, ".", label="numerical")
    plt.xlabel("Amplitude parameter $\epsilon$, 0 < $\epsilon$ <<1")
    plt.ylabel("Superior error for $H^1$ Sobolev norm")
    plt.title(f"$H^1$ norm error for cubic NLKG, $T_0$ ={T}, "+title) 
    plt.plot(x_fit, y_fit, "--", label = "fit = $C\epsilon^b$, with C = {}, b = {}".format(CH, bH) )
    plt.legend(loc = "best")