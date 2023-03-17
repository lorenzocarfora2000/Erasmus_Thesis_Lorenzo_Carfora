import numpy as np
import matplotlib.pyplot as plt
from time import time as clock
from scipy import fft
import the_functions as funcs

k = 0.3    #wavenumber
ac = -1     #cubic nonlinearity parameter

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

Lx = 1000  #absolute value of the x-boundary
dx = 0.02  #imposed spatial step

Nx = int(2*Lx/dx)  #resulting number of points (estimate)


x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx  #resulting x array 
dx = x[1]-x[0]                                #(dx is redefined)

#####################
#TIME EVOLUTION SIMULATOR (RK4)
#####################

t0, tf = 0., 1000      #initial and final time
dt = 0.025            #desired timestep

#soliton initial conditions
"""
env0  = funcs.soliton(e*x, vals_env)
u0    = e*funcs.NLS_approx(x, vals_NLS, funcs.soliton)
v0    = e*funcs.NLS_approx_dt(x, vals, funcs.soliton)
title = "soliton"

"""

# square initial conditions
env0  = funcs.square(e*x, vals_env)
u0    = e*funcs.NLS_approx(x, vals_NLS, funcs.square)
v0    = e*funcs.NLS_approx_dt(x, vals, funcs.square)
title = "square"


if ac== -1: title = "defocusing "+title

#calculation of spatial derivative done via spectral differentiation
#fourier k-space
k1     = 2*np.pi*fft.fftfreq(Nx, d=dx)
k2 = k1**2

K1 = k1/e     #fourier k-space of large X-space
K2 = K1**2

u0_hat = fft.fft(u0) 
u0x    = fft.ifft(1j*k1*u0_hat) 

#Kinetic, potential and strain energies of NLKG. 
#The total energy is their sum
KE0     =   funcs.Lp_norm(v0, 2, x)**2 + funcs.Lp_norm(u0x, 2, x)**2
STRAIN0 =   funcs.Lp_norm(u0, 2, x)**2 
POT0    =   -ac*0.5*funcs.Lp_norm(u0, 4, x)**4

E_TOT0  = KE0 + POT0 + STRAIN0

KE_list     = [KE0]
POT_list    = [POT0]
STRAIN_list = [STRAIN0]
E_TOT_list  = [E_TOT0]

#### NLS approx energy #####

#derivative of envelope over space via spectra differentiation
env0_hat = fft.fft(env0) 
env0x    = fft.ifft(1j*K1*env0_hat) 

E_NLS0 = abs(v2*funcs.Lp_norm(env0x, 2, x)**2 - ac*v3*0.5*funcs.Lp_norm(env0, 4, x)**4) 
E_NLS_list = [E_NLS0] #initial energy

H5_norm_list = [funcs.Hp_norm(2*e*env0, 5, x)] #initial H^5 norm
L2_norm_list = [funcs.Lp_norm(2*e*env0, 2, x)] #initial L2 norm


#start conditions
vals_rk4 = [dx, ac]
vals_strang = [e, c, ac, v1, v2, v3, K1, K2]

t, u, v = t0, u0, v0
env = env0

time = [t0]            #only times were data has been collected)
plot_err_list = [0]    #C^0 error data
plot_err_list_H1 = [0] #H^1 error data


ti = clock()      #start time

superior_error = 0 #biggest error achieved by the approximation

plotgap = 500     #distance between plots
plotnum = 1       #plot number (for initialisation)
n = 0
while t < tf:
    
    #Range-kutta 4 calculation of the NLKG equation
    t, u, v = funcs.RK4(t, u, v, dt, vals_rk4, funcs.NLKG)
    
    #solving the envelope via the NLS equation:
    env = funcs.Strang_splitting(env, dt, vals_strang)
    
    #approximation calculation based on the envelope function
    u_approx = e*env*funcs.Exp(x, t, k, w)   
    u_approx += np.conj(u_approx)
    
    #error calculation
    error = max(abs(u-u_approx))

    if error> superior_error: superior_error = error 

    if (n%plotgap)==0:
        
        H1_error = funcs.Hp_norm(u-u_approx, 1, x) #sobolev norm error
        plot_err_list.append(error)
        plot_err_list_H1.append(H1_error)
        
        profile = 2*e*abs(env)
        
        #plotting numerical evolution over time
        plt.figure(plotnum)
        plt.plot(x, np.real(u),'r-', label = 'Numerical')
        plt.plot(x, np.real(u_approx), "g--", label = 'NLS-approx.')
        plt.plot(x, profile, 'b--', label="Envelope")
        plt.plot(x, -profile, 'b--')
        plt.xlim(c*t-300, c*t + 300)
        plt.legend(loc='best')
        plt.title("Cubic NLKG ("+title+f") $\epsilon$={e}, time t = {np.round(t,3)}")
        plt.xlabel("x")
        plt.ylabel("u(x, t)") 
        plt.show()

        #NLKGE energy calculation:
        u_hat = fft.fft(u) 
        ux    = fft.ifft(1j*k1*u_hat) 
        
        KE     =   funcs.Lp_norm(v, 2, x)**2 + funcs.Lp_norm(ux, 2, x)**2
        STRAIN =   funcs.Lp_norm(u, 2, x)**2 
        POT    =   -ac*0.5*funcs.Lp_norm(u, 4, x)**4
        E_TOT  = KE + POT + STRAIN
    
        
        KE_list.append(KE)     
        POT_list.append(POT)       
        STRAIN_list.append(STRAIN)   
        E_TOT_list.append(E_TOT)    
        time.append(t)
        
        #NLSE consered quantity calculation
        X = e*(x - c*t)
        
        env_hat = fft.fft(env) 
        envx    = fft.ifft(1j*K1*env_hat) 

        E_NLS = abs(v2*funcs.Lp_norm(envx, 2, x)**2 - ac*v3*0.5*funcs.Lp_norm(env, 4, x)**4)
        E_NLS_list.append(E_NLS)
            
        H5_norm_list.append(funcs.Hp_norm(2*e*env, 5, x))
        L2_norm_list.append(funcs.Lp_norm(2*e*env, 2, x))
        
        plotnum+= 1
    n += 1

tf = clock()

print("time taken = {} s".format(tf-ti))

print(f"the superior error is: {superior_error}")


#energy plotting
KE_list = np.array(KE_list).real    
POT_list = np.array(POT_list).real   
STRAIN_list = np.array(STRAIN_list).real     
E_TOT_list = np.array(E_TOT_list).real      

plt.figure(plotnum)
plt.title("Cubic NLKG Energy conservation over time, "+title)
plt.plot(time, KE_list, "b.-",       label = "kinetic" )
plt.plot(time, POT_list , "b--",     label = "potential")
plt.plot(time, STRAIN_list, "g-.",   label = "strain" )
plt.plot(time, E_TOT_list, "r-",    label = "total" )
plt.legend(loc = "best")
plt.xlabel("Time t")
plt.ylabel("Energy")
plt.grid(True)
plt.show()

#logarithmic scale of the energy
E_change = np.log(abs(1-E_TOT_list[1:]/E_TOT0))

plt.figure(plotnum+1)
plt.title("Cubic NLKG Total energy change, "+title)
plt.plot(time[1:], E_change)
plt.grid(True)
plt.xlabel("Time t")
plt.ylabel("Energy change")

time_slow = np.array(time)*(e**2)

plt.figure(plotnum+2)
plt.title("NLS energy of "+title+" envelope")
plt.plot(time_slow, E_NLS_list)
plt.xlabel("Slow time T")
plt.ylabel("Energy")
plt.grid(True)

plt.figure(plotnum+3)
plt.title("$H^5$ Sobolev norm of "+title+" envelope")
plt.plot(time_slow, H5_norm_list)
plt.grid(True)
plt.xlabel("Slow time T")
plt.ylabel("$H^5$ norm")

plt.figure(plotnum+4)
plt.title("$L^2$ norm of "+title+" envelope")
plt.plot(time_slow, L2_norm_list)
plt.grid(True)
plt.xlabel("Slow time T")
plt.ylabel("$L^2$ norm")

"""
plt.figure(plotnum+5)
plt.title("$C^0_b$ and $H^1$ errors of "+title+" solution over time")
plt.plot(time, plot_err_list, label = "$C^0_b$ err")
plt.plot(time, plot_err_list_H1, label = "$H^1$ err")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel("time t")
plt.ylabel("Error value")
"""
