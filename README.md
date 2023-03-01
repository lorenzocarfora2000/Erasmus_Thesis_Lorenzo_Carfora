- the_functions.py : This script contains all the necessary functions for the numerical calculation of the NLKGE and of the NLSE, along with all the functions for the construction of the $\epsilon\psi_{NLS}$ approximation, of the $L^p$ and $H^p$ norms and of the phase correction terms $\Omega_A$ and $\Omega_B$ imposed for the multiplexing case.
- time_evolution.py : This script starts the numerical simulation of the time evolution of the NLKGE. It keeps track of the time evolution by producing plots of the solution at various times and by calculating the conserved NLKGE Hamiltonian $H_{KG}$. The code also simulates the $\epsilon\psi_{NLS}$ approximation so to compare it to the NLKGE numerical results. The NLSE time evolution is also kept under check by calculating the NLSE conserved Mass $M_{NLS}$ and Hamiltonian $H_{NLS}$.
- multiplexing.py : The script has a similar use to the "time$\_$evolution.py" script, only it is used to simulate the numerical time evolution of two multiplexing carrier waves.
- superior_error_behaviour.py : This script calculates the relation between the superior error of the approximation and the amplitude parameter $\epsilon$. The code runs a time evolution of the NLKGE and compares it to $\epsilon\psi_{NLS}$, retrieving the corresponding superior error. This procedure is then repeated for different $\epsilon$ parameters, allowing for the numerical evaluation of the efficiency of the approximation. For better evaluations, the superior error is retrieved considering both the $C_b^0$ and the $H^1$ norms.
- superior_error_multiplex.py : similar to superior$\_$error$\_$behaviour.py, this script is used to analyse the approximation efficiency in the case of two multiplexing carrier waves. The script considers the superior error behaviour by evaluating both the $C_b^0$ and the $H^1$ norms, and it does so for both cases including and excluding the phase corrections $\Omega_A$ and $\Omega_B$.
- dx_consistency.py : This script is used to check the consistency of the superior error behaviour when considering the spatial discretisations of the considered numerical evaluations. The script calculates the parameters of the $C\epsilon^b$ function fitting the relation between the superior error of the approximation - considering both the $C_b^0$ and $H^1$ norms - and the applied $\epsilon$ value. This procedure is done for different values of $dx$ so to obtain a better observation of how would the superior error behaviour depend on the numerical discretisation. Given the linear behaviour observed for the $b$ parameter, a linear fit was done to estimate what would the parameter be for the continuous case $dx\rightarrow 0$.