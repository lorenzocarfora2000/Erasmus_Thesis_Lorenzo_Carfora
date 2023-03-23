import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time as clock
import multiprocessing as mp
import the_functions as funcs
from scipy import fft

k = np.array([0.9, -1])  #Wavenumbers
w = np.sqrt(k**2 + 1)    #Frequencies 
c = k/w                  #Group velocities

ac = 1                   #cubic nonlinearity sign
e = 0.1                  #Amplitude (0< e << 1)

#NLSE parameters
gamma = 0.5
v1, v2, v3 = 2*w, (1-c**2) ,  3
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

#Initial displacements
x0 = [-55, 55]
x0_A, x0_B = x0[0], x0[1]

##Spatial array
Lx = 1000
dx = 0.02

Nx =int(2*Lx/dx)

x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx
dx = x[1]-x[0]

T = 1.4
dt = 0.015

dx = x[1]-x[0]

#Wavenumber array for Fourier transformations
k1 = 2*np.pi*fft.fftfreq(Nx, d=dx)

#The procedure is the same as in superior_error_behaviour.py, only adapted for 
#the multiplexing case. The sup_err_calc function is also modified to return
#the C^0_b and H^1 norms of the error deviation for both the cases with 
#and without phase correction.

vals_rk4 = [dx, ac]
def sup_err_calc(e):
    
    vals_A = A[0], B[0], gamma, e, k[0], w[0], c[0], ac, v1[0], v2[0], v3
    vals_B = A[1], B[1], gamma, e, k[1], w[1], c[1], ac, v1[1], v2[1], v3
    
    vals_env_A, vals_NLS_A = vals_A[:3], vals_A[:7]
    vals_env_B, vals_NLS_B = vals_B[:3], vals_B[:7]
    
    env_A  = funcs.soliton(e*(x-x0_A), vals_env_A)
    u_A0    = e*funcs.NLS_approx(x-x0_A, vals_NLS_A, funcs.soliton)
    v_A0    = e*funcs.NLS_approx_dt(x-x0_A, vals_A, funcs.soliton)

    # square initial conditions
    env_B  = funcs.square(e*(x-x0_B), vals_env_B)
    u_B0    = e*funcs.NLS_approx(x-x0_B, vals_NLS_B, funcs.square)
    v_B0    = e*funcs.NLS_approx_dt(x-x0_B, vals_B, funcs.square)

    u0 = u_A0+u_B0
    v0 = v_A0+v_B0 
    
    t, u, v, tf = 0., u0, v0, T/(e**2)
    
    K1 = k1/e
    K2 = K1**2
    
    vals_strang_A = [e, c[0], ac, v1[0], v2[0], v3, K1, K2]
    vals_strang_B = [e, c[1], ac, v1[1], v2[1], v3, K1, K2]
    
    superior_error = 0
    superior_H_error = 0
    
    superior_error_shift = 0
    superior_H_error_shift = 0
    
    while t < tf:

        #runge kutta 4 calculation of the NLKG equation
        t, u, v = funcs.RK4(t, u, v, dt, vals_rk4, funcs.NLKG)
        
        
        env_A = funcs.Strang_splitting(env_A, dt, vals_strang_A)
        env_B = funcs.Strang_splitting(env_B, dt, vals_strang_B)
        
        if t < 100: continue
    
        Phase_A, Phase_B = funcs.phase_shifts(c, w, e, env_A, env_B, dx)
        #Phase_A, Phase_B   = funcs.phase_shifts_soliton(x, t, c, w, e, A, B, x0)
        
        env_A_approx = e*(env_A*funcs.Exp(x-x0_A, t, k[0], w[0]))
        env_B_approx = e*(env_B*funcs.Exp(x-x0_B, t, k[1], w[1]))

        #approximation calculation based on the envelope function
        u_approx = env_A_approx + env_B_approx
        
        u_approx_shift = env_A_approx*np.exp(1j*e*Phase_A) + env_B_approx*np.exp(1j*e*Phase_B)
        
        u_approx += np.conj(u_approx)
        u_approx_shift += np.conj(u_approx_shift)
        
        
        error = max(abs(u-u_approx))
        H_error = funcs.Hp_norm(u-u_approx, 1, x) #sobolev norm error
        
        error_shift = max(abs(u-u_approx_shift))
        H_error_shift = funcs.Hp_norm(u-u_approx_shift, 1, x) #sobolev norm error
    
        if error> superior_error: superior_error = error 
        if H_error> superior_H_error: superior_H_error = H_error 
        
        if error_shift> superior_error_shift: superior_error_shift = error_shift 
        if H_error_shift> superior_H_error_shift: superior_H_error_shift = H_error_shift 
        
    print(f"done with {e}")
    return superior_error, superior_H_error, superior_error_shift, superior_H_error_shift

#list of amplitude parameters
e_list = np.linspace(0.07, 0.1, 4)

if __name__ == '__main__':
    
    start = clock()
    
    #Parallel calculation, sup_err_set contains all the results for all the
    #e_list parameters
    p = mp.Pool()
    sup_err_set = p.map(sup_err_calc,  e_list)
    p.close()
    p.join()
    end = clock()
    
    #Separating the results derivated results over e_list in lists
    sup_err_list, sup_H_err_list, sup_err_list_shift, sup_H_err_list_shift = np.array(sup_err_set).T
    
    print(f"time taken: {end-start} seconds")
    
    #Append the trivial solution of e = 0.
    e_list = np.append(0, e_list)
    sup_err_list = np.append(0, sup_err_list)
    sup_H_err_list = np.append(0, sup_H_err_list)
    sup_err_list_shift = np.append(0, sup_err_list_shift)
    sup_H_err_list_shift = np.append(0, sup_H_err_list_shift)

    fitter = lambda x, a, b: a*x**b   #Fitting function 

    #Fitting and plotting for the error in C^0 (no correction)
    popt, pcov = curve_fit(fitter, e_list, sup_err_list)
    x_fit = np.linspace(0, e_list[-1], 100)
    y_fit = fitter(x_fit, *popt)
    
    #Also plotting for the error in C^0 (with correction)
    popt_shift, pcov_shift = curve_fit(fitter, e_list, sup_err_list_shift)
    y_fit_shift = fitter(x_fit, *popt_shift)
    
    
    C, b = np.round(popt, 2)
    C_shift, b_shift = np.round(popt_shift, 2)

    plt.figure(3)
    plt.plot(e_list, sup_err_list, ".", label="No Correction")
    plt.plot(e_list, sup_err_list_shift, ".", label="Correction")
    plt.xlabel("Amplitude parameter, 0 < $\epsilon$ <<1")
    plt.ylabel("$C_b^0$ norm of superior error")
    plt.title(f"Multiplexing error consistency, cubic NLKG, T$_0$ ={T}") 
    plt.plot(x_fit, y_fit, "--", label = f"No correction fit, $\epsilon^b$, b={b}" )
    plt.plot(x_fit, y_fit_shift, "--", label = f"Correction fit, $\epsilon^b$, b={b_shift}")
    plt.legend(loc = "best")
    
    
    #Fitting and plotting for the error in H^1 (no correction)
    poptH, pcovH = curve_fit(fitter, e_list, sup_H_err_list)
    y_fit = fitter(x_fit, *poptH)
    
    #Also plotting for the error in H^1 (with correction)
    poptH_shift, pcovH_shift = curve_fit(fitter, e_list, sup_H_err_list_shift)
    y_fit_shift = fitter(x_fit, *poptH_shift)

    CH, bH = np.round(poptH, 2)
    CH_shift, bH_shift = np.round(poptH_shift, 2)

    plt.figure(4)
    plt.plot(e_list, sup_H_err_list, ".", label="No Correction")
    plt.plot(e_list, sup_H_err_list_shift, ".", label="Correction")
    plt.xlabel("Amplitude parameter, 0 < $\epsilon$ <<1")
    plt.ylabel("$H^1$ norm of superior error")
    plt.title(f"Multiplexing $H^1$ norm error consistency, cubic NLKG, T$_0$ ={T}") 
    plt.plot(x_fit, y_fit, "--", label = f"No correction fit, $\epsilon^b$, b={bH}" )
    plt.plot(x_fit, y_fit_shift, "--", label = f"Correction fit, $\epsilon^b$, b={bH_shift}")
    plt.legend(loc = "best")
