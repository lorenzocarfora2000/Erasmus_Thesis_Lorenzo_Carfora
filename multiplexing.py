import numpy as np
import matplotlib.pyplot as plt
from time import time as clock
from scipy import fft
import the_functions as funcs

k = np.array([0.9, -1])     #Wavenumebers
w = np.sqrt(k**2 + 1)       #Frequencies 
c = k/w                     #Group velocities

ac = 1                      #Cubic nonlinearity sign
e = 0.1                     #Amplitude (0< e << 1)

#NLSE parameters
gamma = 0.5
v1, v2, v3 = 2*w, (1-c**2) ,  3
A, B = np.sqrt(2*gamma*v1/v3) , np.sqrt(gamma*v1/v2) 

#Reorder in arrays for pulse A and for pulse B
vals_A = A[0], B[0], gamma, e, k[0], w[0], c[0], ac, v1[0], v2[0], v3
vals_B = A[1], B[1], gamma, e, k[1], w[1], c[1], ac, v1[1], v2[1], v3

vals_env_A, vals_NLS_A = vals_A[:3], vals_A[:7]
vals_env_B, vals_NLS_B = vals_B[:3], vals_B[:7]

##Spatial array
Lx = 900
dx = 0.02

Nx =int(2*Lx/dx)

x = 2*Lx*np.arange(-int(Nx/2), int(Nx/2))/Nx
dx = x[1]-x[0]

#Initial displacements
x0 = [-60, 60]                
x0_A, x0_B = x0[0], x0[1]

#Soliton initial conditions
env_A0  = funcs.soliton(e*(x-x0_A), vals_env_A)
u_A0    = e*funcs.NLS_approx(x-x0_A, vals_NLS_A, funcs.soliton)
v_A0    = e*funcs.NLS_approx_dt(x-x0_A, vals_A, funcs.soliton)

#Square initial conditions
env_B0  = funcs.square(e*(x-x0_B), vals_env_B)
u_B0    = e*funcs.NLS_approx(x-x0_B, vals_NLS_B, funcs.square)
v_B0    = e*funcs.NLS_approx_dt(x-x0_B, vals_B, funcs.square)

#Total initial condition solutions
u0 = u_A0+u_B0
v0 = v_A0+v_B0 

#Fourier space setup
k1 = 2*np.pi*fft.fftfreq(Nx, d=dx)
k2 = k1**2

K1 = k1/e
K2 = K1**2

vals_rk4 = [dx, ac]
vals_strang_A = [e, c[0], ac, v1[0], v2[0], v3, K1, K2]
vals_strang_B = [e, c[1], ac, v1[1], v2[1], v3, K1, K2]

#Start conditions
plotnum = 1       #plot number
plotgap = 200     #distance between plots

#####################
#TIME EVOLUTION SIMULATOR (RK4)
#####################

t0, tf = 0., 170       #Initial and final time
dt = 0.025             #Desired timestep

#Biggest error over space achieved by the approximation at different times

plot_err_list = [0]         #With no correction
plot_err_list_shift = [0]   #With correction
time = [t0]

t, u, v = t0, u0 , v0
env_A = env_A0
env_B = env_B0

superior_error = 0
superior_error_shift = 0

ti = clock()

n = 0
while t < tf:
    
    #Runge-kutta 4 calculation of the NLKG equation
    t, u, v = funcs.RK4(t, u, v, dt, vals_rk4, funcs.NLKG)
    
    #Strang splitting for envelopes A and B
    env_A = funcs.Strang_splitting(env_A, dt, vals_strang_A)
    env_B = funcs.Strang_splitting(env_B, dt, vals_strang_B)
    
    env_A_approx = e*(env_A*funcs.Exp(x-x0_A, t, k[0], w[0]))
    env_B_approx = e*(env_B*funcs.Exp(x-x0_B, t, k[1], w[1]))
    
    #approximation without shift correction
    u_approx = env_A_approx + env_B_approx
    u_approx += np.conj(u_approx)
    
    #Phase shift corrections for pulses A and B
    Phase_A, Phase_B = funcs.phase_shifts(c, w, e, env_A, env_B, dx)
    
    #Approximation with shift correction
    u_approx_shift = env_A_approx*np.exp(1j*e*Phase_A) + env_B_approx*np.exp(1j*e*Phase_B)
    u_approx_shift += np.conj(u_approx_shift)
    
    #Plotting of approximation errors 
   
    error = max(abs(u-u_approx))
    error_shift = max(abs(u-u_approx_shift))
    
    if error> superior_error: superior_error = error 
    if error_shift> superior_error_shift: superior_error_shift = error_shift 
    
    if (n%plotgap)==0:

        env = env_A + env_B
        profile = 2*e*abs(env)
        
        plt.figure(plotnum)
        plt.plot(x, np.real(u),'r-', label = 'numerical')
        plt.plot(x, np.real(u_approx_shift), "g--", label = 'NLS-approx')
        plt.plot(x, profile, 'b--', label="envelope")
        plt.plot(x, -profile, 'b--')
        plt.plot(x, e*Phase_B, "--", label= "Phase for A")
        plt.plot(x, e*Phase_A, "--", label = "Phase for B")
        plt.xlim(-200,  200)
        plt.legend(loc="best")
        plt.title(f"Cubic NLKG (multiplexing), $\epsilon$={e}, time t = {np.round(t,3)}")
        plt.xlabel("x")
        plt.ylabel("u(x, t)") 
        plt.show()
        
        time.append(t)
        plot_err_list.append(error)
        plot_err_list_shift.append(error_shift)
        
        plotnum+= 1
    n += 1

tf = clock()

print(f"time taken = {tf-ti} s")
print(max(plot_err_list_shift))


plt.figure(plotnum+2)
plt.title("$C^0_b$ errors over time for Interacting Carrier waves")
plt.plot(time, plot_err_list, ".-", label = "no Correction")
plt.plot(time, plot_err_list_shift, ".-", label = "Correction")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel("time t")
plt.ylabel("error value")



