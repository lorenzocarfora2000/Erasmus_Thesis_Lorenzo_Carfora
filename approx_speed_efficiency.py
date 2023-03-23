import numpy as np
import matplotlib.pyplot as plt
from time import time as clock
from scipy import fft
import the_functions as funcs
import multiprocessing as mp

k = 0.3    #Wavenumber
ac = 1     #Cubic nonlinearity sign

e = 0.1    #Amplitude (0< e << 1)

#NLS parameters
w = np.sqrt(k**2 + 1)    #Frequency 
c = k/w                  #Group velocity

#Constants defining the equation
v1, v2, v3 = 2*w, (1-c**2) ,  3

#Parameters for the initial condition solution
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

#Collecting into arrays
vals = [A, B, gamma, e, k, w, c, ac, v1, v2, v3]
vals_env = vals[:3]
vals_NLS = vals[:7]

#The speed_measurer function measures the time taken for the NLKGE and NLSE 
#approximation routines to calculate the final results for a given number of
#data points N
def speed_measurer(N):
    
    N = int(N)
    
    #"Large" spatial array
    LX = 100                                  #Absolute value of the x-boundary
    X = 2*LX*np.arange(-int(N/2), int(N/2))/N #Resulting x array 
    dX = X[1]-X[0]                            #dx is redefined

    #####################
    #NLSE approximation routine
    #####################

    T0, Tf = 0., 10      #Initial and final "slow" time
    dT = 0.005           #desired timestep

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
    
    #Strang splitting routine
    clock_i = clock()
    while T < Tf:
        env = funcs.Strang_splitting_large_frame(env, dT, vals_strang)
        T += dT

    clock_f = clock()
    
    time_NLS = clock_f-clock_i #Time taken for the NLSE approximation
    
    #Convertion to standard variables
    x_NLS = (X + c*T/e)/e
    t = T/(e**2)
    
    #Constructing the approximation
    u_approx = e*env*funcs.Exp(x_NLS, t, k, w)   
    u_approx += np.conj(u_approx)

    #####################
    #NLSE approximation routine
    #####################     
    
    #corresponding x-array and final time in standard variables
    x = X/e  
    dx = x[1]-x[0]

    t0, tf = 0. , Tf/(e**2)      #initial and final time
    dt = 0.05 

    t, u, v = t0, u0, v0
    vals_rk4 = [dx, ac]
    
    clock_i = clock()
    
    #RK4 routine
    while t < tf:
         t, u, v = funcs.RK4(t, u, v, dt, vals_rk4, funcs.NLKG)
    clock_f = clock()

    time_NLKG = clock_f-clock_i
    
    #Plots of the profile of the final results of the NLSE and NLKGE to
    #assure the two results are consistent
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"for T = {Tf}, N = {N}")
    ax[0].plot(x_NLS, np.real(u_approx))
    ax[0].set( xlim=(-100+c*t , 100 + c*t), xlabel="standard space x" , ylabel="u(x, t)", title="NLS approximation solution")
    
    ax[1].plot(x, np.real(u))
    ax[1].set( xlim=(-100+c*t , 100 + c*t), xlabel="standard space x" , ylabel="u(x, t)", title= "NLKG solution")
    fig.tight_layout()
    
    return time_NLS , time_NLKG, fig

#List of given number of points
N_arr = np.linspace(5e3, 25e3, 9, dtype= int)

if __name__ == '__main__':
    
    #Parallel calculation of the time taken by the routines for different 
    #numbers of data points
    p = mp.Pool()
    fitting = p.map(speed_measurer, N_arr)
    p.close()
    p.join()
    
    time_NLS_arr, time_NLKG_arr, a = np.array(fitting).T
    
    #Plotting the results
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



