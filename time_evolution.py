import numpy as np
import matplotlib.pyplot as plt
from time import time as clock
from scipy import fft
import the_functions as funcs

k = 0.3                  #Wavenumber
ac = +1                  #Cubic nonlinearity sign
e = 0.1                  #Amplitude (0< e << 1)

#NLSE parameters:
w = np.sqrt(k**2 + 1)    #Frequency 
c = k/w                  #Group velocity

#Constants defining the NLSE:
v1, v2, v3 = 2*w, (1-c**2) ,  3

#Parameters for the initial condition solution:
gamma = 0.5
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

#Collecting into arrays:
vals = [A, B, gamma, e, k, w, c, ac, v1, v2, v3]
vals_env = vals[:3]
vals_NLS = vals[:7]

#Creating the spatial array:

Lx = 1000                      #Absolute value of the x-boundary
dx = 0.02                      #Imposed (initial) spatial step

Nx = int(2*Lx/dx)              #Resulting number of points (estimate)


x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx    #Resulting x array 
dx = x[1]-x[0]                                  #dx is redefined

#Soliton initial conditions
"""
env0  = funcs.soliton(e*x, vals_env)
u0    = e*funcs.NLS_approx(x, vals_NLS, funcs.soliton)
v0    = e*funcs.NLS_approx_dt(x, vals, funcs.soliton)
title = "soliton"
"""

#Square initial conditions
env0  = funcs.square(e*x, vals_env)
u0    = e*funcs.NLS_approx(x, vals_NLS, funcs.square)
v0    = e*funcs.NLS_approx_dt(x, vals, funcs.square)
title = "square"

if ac== -1: title = "defocusing "+title

####Calculation of spatial derivative via spectral differentiation #######
#Fourier k-space
k1     = 2*np.pi*fft.fftfreq(Nx, d=dx)
k2 = k1**2

#Fourier K-space of large X-space
K1 = k1/e         
K2 = K1**2

#Space derivative via Fourier transformation 
u0_hat = fft.fft(u0) 
u0x    = fft.ifft(1j*k1*u0_hat) 

###### NLKGE energy ######
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
E_NLS_list = [E_NLS0]     #initial energy

H5_norm_list = [funcs.Hp_norm(2*e*env0, 5, x)] #initial H^5 norm
L2_norm_list = [funcs.Lp_norm(2*e*env0, 2, x)] #initial L2 norm

################################
#TIME EVOLUTION SIMULATOR (RK4)
################################

t0, tf = 0., 1000      #Initial and final time
dt = 0.025             #Desired timestep

#Start conditions
vals_rk4 = [dx, ac]
vals_strang = [e, c, ac, v1, v2, v3, K1, K2]

t, u, v = t0, u0, v0
env = env0

time = [t0]               #Only times were data has been collected)

ti = clock()              #Start measuring time

superior_error = 0        #Biggest error of the approximation

plotgap = 500             #Distance between plots
plotnum = 1               #Plot number (for initialisation)

n = 0
while t < tf:
    
    #Range-kutta 4 calculation of the NLKG equation
    t, u, v = funcs.RK4(t, u, v, dt, vals_rk4, funcs.NLKG)
    
    #Solving the envelope via the Strang splitting of the NLS equation:
    env = funcs.Strang_splitting(env, dt, vals_strang)
    
    #Approximation calculation based on the envelope function
    u_approx = e*env*funcs.Exp(x, t, k, w)   
    u_approx += np.conj(u_approx)
    
    #Error calculation
    error = max(abs(u-u_approx))

    if error> superior_error: superior_error = error 

    if (n%plotgap)==0:
        
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
        
        #NLSE consered quantities and H^5 norm calculation
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

print(f"time taken = {tf-ti} s")

print(f"the superior error is: {superior_error}")

#Energy plotting
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

#Logarithmic scale of the energy
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