import numpy as np
import matplotlib.pyplot as plt
from time import time as clock
from scipy import fft
import the_functions as funcs
import multiprocessing as mp

k = 0.3    #wavenumber
ac = 1     #cubic nonlinearity parameter

e = 0.1    #amplitude (0< e << 1)

#NLS parameters
w = np.sqrt(k**2 + 1)    #frequency 
c = k/w                  #group velocity

#constants defining the equation
v1, v2, v3 = 2*w, (1-c**2) ,  3

#parameters for the initial condition solution
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

#collecting into arrays
vals = [A, B, gamma, e, k, w, c, ac, v1, v2, v3]
vals_env = vals[:3]
vals_NLS = vals[:7]

## spatial array

def speed_measurer(N):
    N = int(N)
    
    LX = 100 #absolute value of the x-boundary
    X = 2*LX*np.arange(-int(N/2), int(N/2))/N  #resulting x array 
    dX = X[1]-X[0]                                #(dx is redefined)

    #####################
    #TIME EVOLUTION SIMULATOR (RK4)
    #####################

    T0, Tf = 0., 10      #initial and final time
    dT = 0.005            #desired timestep

    #soliton initial conditions

    env0  = funcs.soliton(X, vals_env) 
    u0    = e*funcs.NLS_approx(X/e, vals_NLS, funcs.soliton) 
    v0    = e*funcs.NLS_approx_dt(X/e, vals, funcs.soliton)


    #calculation of spatial derivative done via spectral differentiation 
    #fourier k-space
    K1 = 2*np.pi*fft.fftfreq(N, d=dX)
    K2 = K1**2

    vals_strang = [ac, v1, v2, v3, K1, K2]

    T, env = T0, env0

    clock_i = clock()
    while T < Tf:
        env = funcs.Strang_splitting_large_frame(env, dT, vals_strang)
        T += dT

    clock_f = clock()
    
    time_NLS = clock_f-clock_i

    x_NLS = (X + c*T/e)/e
    t = T/(e**2)

    u_approx = e*env*funcs.Exp(x_NLS, t, k, w)   
    u_approx += np.conj(u_approx)

    x = X/e  
    dx = x[1]-x[0]

    t0, tf = 0. , Tf/(e**2)      #initial and final time
    dt = 0.05 

    t, u, v = t0, u0, v0
    vals_rk4 = [dx, ac]

    clock_i = clock()
    while t < tf:
         t, u, v = funcs.RK4(t, u, v, dt, vals_rk4, funcs.NLKG)
    clock_f = clock()

    time_NLKG = clock_f-clock_i
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"for T = {Tf}, N = {N}")
    ax[0].plot(x_NLS, np.real(u_approx))
    ax[0].set( xlim=(-100+c*t , 100 + c*t), xlabel="standard space x" , ylabel="u(x, t)", title="NLS approximation solution")
    
    ax[1].plot(x, np.real(u))
    ax[1].set( xlim=(-100+c*t , 100 + c*t), xlabel="standard space x" , ylabel="u(x, t)", title= "NLKG solution")
    fig.tight_layout()
    
    return time_NLS , time_NLKG, fig

N_arr = np.linspace(5e3, 25e3, 9, dtype= int)

if __name__ == '__main__':
    
    p = mp.Pool()
    fitting = p.map(speed_measurer, N_arr)
    p.close()
    p.join()
    
    time_NLS_arr, time_NLKG_arr, a = np.array(fitting).T
    
    plt.figure(10)
    plt.plot(N_arr, time_NLS_arr, ".-", label = "Time for NLSE calculations")
    plt.plot(N_arr, time_NLKG_arr, ".-", label = "Time for NLKGE calculations")
    plt.legend(loc="best")
    plt.xlabel("Number of data points N")
    plt.ylabel("Time taken (s)")
    plt.title("Time efficiency of NLS and NLKG routines")    
    plt.grid(True)
    
    ratio = np.mean(time_NLKG_arr/time_NLS_arr)
    print(f"Average time ratio: NLSE routine is {ratio} times faster than the NLKGE one")



