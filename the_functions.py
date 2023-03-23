import numpy as np
from scipy import fft
from scipy.integrate import trapz

#Exponential function
Exp = lambda x, t, k, w : np.exp(1j*(k*x - w*t))

#Hyperbolic secant function
sech = lambda x : 1/np.cosh(x)

#Soliton solution initial condition for NLSE
def soliton(X, vals):
    A, B, gamma = vals
    return A*sech(B*X)

#Square solution initial condition for NLSE
def square(X, vals):
    A, B, gamma = vals
    return 0.25*A*(1-np.tanh(B*(X-11*gamma))*np.tanh(B*(X+11*gamma)))


#NLKGE equation approximation using the initial conditions:
def NLS_approx(x, vals, envelope):
    
    #Needed constants
    A, B, gamma, e, k, w, c = vals
    
    #Defining slow space coords
    X = e*x
    
    env = envelope(X, [A, B, gamma])
    f = env*Exp(x, 0, k, w)
    
    return f + np.conj(f) 

#Time derivative of the NLKGE approximation via Fourier transformation
def NLS_approx_dt(x, vals, envelope):
    
    #needed constants
    A, B, gamma, e, k, w, c, ac, v1, v2, v3 = vals
    
    #slow space and time coords
    X = e*x
    
    ### first and second derivative of the envelope via Fourier transform
    env = envelope(X, [A, B, gamma])
    env_hat = np.fft.fft(env)
    
    k1 = 2*np.pi*np.fft.fftfreq(len(x), (x[1]-x[0]))
    k2 = k**2
    
    env_x_hat, env_xx_hat = 1j*k1*env_hat, -k2*env_hat
    env_x = e*np.real(np.fft.ifft(env_x_hat))
    env_xx =  (e**2)*np.real(np.fft.ifft(env_xx_hat))
    
    #the approximation
    U = Exp(x, 0, k, w)*(-1j*w*env - e*c*env_x +
                         1j*(v2*env_xx + ac*v3*env*abs(env)**2)*(e**2)/v1  )

    return U + np.conj(U) 

#L_p norm, requires function u, degree p and xarray x
def Lp_norm(u, p, x): return trapz( abs(u)**p , x)**(1/p)
    
#Sobolev H_p norm
def Hp_norm(u, p, x):
    Nx, dx = len(x), x[1]-x[0] 
    
    u_hat = fft.fft(u)
    kappa = 2*np.pi*fft.fftfreq(Nx, d=dx)
    
    I = 0
    for n in range(p+1):
        
        if n == 0: I += Lp_norm(u, 2, x)**2
        
        else:
            du_hat = ((1j*kappa)**n)*u_hat
            du = fft.ifft(du_hat)
            I += Lp_norm(du, 2, x)**2
            
    return np.sqrt(I)   

#Second spatial derivative function, using the discretisation in physical space
def Del(u, dx): 
    return (np.roll(u, -1) -2*u + np.roll(u, 1))/dx**2

#Nonlinear cubic Klein gordon equation:
def NLKG(u, vals):
    dx, ac = vals
    dvdt = Del(u, dx) - u + ac*u**3
    return dvdt

#Rk4 routine for the NLKGE:
def RK4(t, u, v, dt, vals, function):
    
    #RK4 routine
    k1_v = dt*function(u, vals)
    k1_x = dt*v
    k2_v = dt*function(u+k1_x/2, vals)
    k2_x = dt*(v+k1_v/2)
    k3_v = dt*function(u +k2_x/2, vals)
    k3_x = dt*(v + k2_v/2)
    k4_v = dt*function(u + k3_x, vals)
    k4_x = dt*(v + k3_v)

    vnew = v + k1_v/6 + k2_v/3 + k3_v/3 + k4_v/6 #new v and pos vals
    unew = u +k1_x/6 + k2_x/3 + k3_x/3 + k4_x/6
    tnew = t+dt
    return tnew, unew, vnew

#Routine for the NLSE calculation
#(input the initial condition in w.r.t X, dt and vals. K1, K2 are the 
#space wavelength numbers w.r.t X.)
def Strang_splitting(U, dt, vals):
    
    #Needed constants
    e, c, ac, v1, v2, v3, K1, K2 = vals
    
    #the NLSE is expressed w.r.t T-scale, to convert to t-scale:
    C  = c/e 
    dT = (e**2)*dt
    
    #Solving map 1 for half time step:
    U_hat = fft.fft(U)
    w_hat = np.exp(-1j*(K2*(v2/v1)+C*K1)*0.5*dT)*U_hat 
    w     = fft.ifft(w_hat)
    
    #Solving map 2 for full time step:
    wnew = np.exp(1j*ac*(v3/v1)*dT*(abs(w))**2)*w
    
    #Solving map 1 for half time step again:
    wnew_hat = fft.fft(wnew)
    Unew_hat = np.exp(-1j*(K2*(v2/v1)+C*K1)*0.5*dT)*wnew_hat
    
    Unew     = fft.ifft(Unew_hat) 
    return Unew

#Strang splitting routine as before, only it accepts data coded to be 
#representing the large frame values. It is set in the comoving frame of 
#reference and returns the results in terms of X and T.
def Strang_splitting_large_frame(U, dT, vals):
    
    #Needed constants
    ac, v1, v2, v3, K1, K2 = vals

    #Solving map 1 for half time step:
    U_hat = fft.fft(U)
    w_hat = np.exp(-1j*K2*(v2/v1)*0.5*dT)*U_hat 
    w  = fft.ifft(w_hat)
    
    #Solving map 2 for full time step:
    wnew = np.exp(1j*ac*(v3/v1)*dT*(abs(w))**2)*w
    
    #Solving map 1 for half time step again:
    wnew_hat = fft.fft(wnew)
    Unew_hat = np.exp(-1j*K2*(v2/v1)*0.5*dT)*wnew_hat
    
    Unew     = fft.ifft(Unew_hat) 
    return Unew


#Phase shift correction for carrier waves interaction study:
def phase_shifts(c, w, e, env_A0, env_B0, dx):
    
    #The constants characterising the the phase shifts of pulses A and B:
    C = 3/(w*(c - np.flip(c)))
    
    #Envelope reordering, taking the absolute value squared:
    envs = np.vstack([env_A0, env_B0])
    absolute = abs(envs)**2
    
    #Calculation of the area under under the function for each step taken.
    integral = (absolute[:, :-1] + absolute[:, 1:])*dx/2
    
    #the areas approximately equal to zero are set directly to zero
    integral = np.where(integral<1e-9, 0, integral )
    
    #Since pulses are localised, we expect that the given array is already
    #close to zero at the left boundary (limit for -inf):
    integral = np.hstack([np.array([[0], [0]]), integral])
    
    #Find position of values that are big enough to be considered
    nonzeros_pos = np.where(integral != 0) 
    
    #Positions of values for the single A and B pulses
    pos_A = nonzeros_pos[1][np.where(nonzeros_pos[0]==0)]
    pos_B = nonzeros_pos[1][np.where(nonzeros_pos[0]==1)]
    
    #Putting the pulses A and B values in separate arrays:
    nonzeros_A = integral[0, pos_A]
    nonzeros_B = integral[1, pos_B]
    
    #Boundaries of the effective domains of localised A and B
    lm_A, lp_A = pos_A[0], pos_A[-1]
    lm_B, lp_B = pos_B[0], pos_B[-1]
    
    #Calculate integral result for the influencing area
    result_A, result_B = np.copy(nonzeros_A), np.copy(nonzeros_B)
    for i in range(len(result_A)):
        result_A[i] = np.sum(nonzeros_A[:(i+1)])
    for i in range(len(result_B)):
        result_B[i] = np.sum(nonzeros_B[:(i+1)])
        
    integral[0, pos_A] = result_A
    integral[1, pos_B] = result_B
    
    #Null spaces could be present in between the values, as dips in functions
    #describing A and B may occur: the following routine corrects that:
    A, B = integral[0, lm_A:lp_A],  integral[1, lm_B:lp_B]
    if bool(A.all()) == False:
        while np.any(A==0):
            indices=np.where(A==0)
            newvalueindices=np.subtract(indices,np.ones_like(indices))
            A[indices]=A[newvalueindices]
    
    if bool(B.all()) == False:
        while np.any(B==0):
            indices=np.where(B==0)
            newvalueindices=np.subtract(indices,np.ones_like(indices))
            B[indices]=B[newvalueindices]
    
    integral[0, lm_A:lp_A] = A
    integral[1, lm_B:lp_B] = B
    
    #The uninfluencing parts of the integral stay zero before,   
    #and remain with the same result as the last calculated one after
    integral[0, pos_A[-1]:] = result_A[-1]
    integral[1, pos_B[-1]:] = result_B[-1]
    
    #Row 0 is the integral of |A|^2, row 1 is the integral of |B^2|
    
    #The integral is multiplied by the constants accordingly to be equal to 
    #the phase shifts of A and B
    integral[0] *= e*C[1]
    integral[1] *= e*C[0]
    
    #Shifting for phase correction normalisation
    integral[0] -= np.min(integral[0])
    integral[1] -= np.min(integral[1])
        
    return integral[1], integral[0]
