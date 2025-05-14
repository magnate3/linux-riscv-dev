from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def EulerForward(func, t_span, y0, n):
    """ Explicit Euler method
    Parameters
    ----------
    func : function, function handler of the right hand side of the ODE(s);
    t_span : list, t_span[0] is the initial time, t_span[1] is the final time;
    y0 : float, list, initial condition(s) of the ODE(s);
    n : number of time steps, dt = (t_span[1]-t_span[0])/n.
        
    Returns
    -------
    t : list, the resulting time steps;
    y : list, the solutions of the ODE(s)
    """
    t = np.linspace(t_span[0], t_span[1], n)
    dt = t[1]-t[0]
    y = [y0]
    for i in range(n-1):
        ynp1 = y[i] + dt*func(t[i], y[i])
        y.append(ynp1)
    return t, y
    
    
def Heun(func, t_span, y0, n):
    """ Heun's method
    Parameters
    ----------
    func : function, function handler of the right hand side of the ODE(s);
    t_span : list, t_span[0] is the initial time, t_span[1] is the final time;
    y0 : float, list, initial condition(s) of the ODE(s);
    n : number of time steps, dt = (t_span[1]-t_span[0])/n.
        
    Returns
    -------
    t : list, the resulting time steps;
    y : list, the solutions of the ODE(s)
    """
    t = np.linspace(t_span[0], t_span[1], n)
    dt = t[1]-t[0]
    y = [y0]
    for i in range(n-1):
        k1 = func(t[i], y[i])
        k2 = func(t[i+1], y[i]+dt*k1)
        ynp1 = y[i] + 0.5*dt*(k1+k2)
        y.append(ynp1)
    return t, y
def rhsODEs(t, y):
    return y - 0.5*np.exp(0.5*t)*np.sin(5*t)+5*np.exp(0.5*t)*np.cos(5*t)

# initial condition
y0 = [0]
N = 20

# Time steps
t_span = (0, np.pi)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve for the ODE with R-K method
sol_ex = solve_ivp(rhsODEs, t_span, y0, method='RK45', t_eval=t_eval)
sol_fe = EulerForward(rhsODEs, t_span, y0, N)
sol_he = Heun(rhsODEs, t_span, y0, N)
t_evalRK = np.linspace(t_span[0], t_span[1], N)
sol_rk = solve_ivp(rhsODEs, t_span, y0, method='RK45', t_eval=t_evalRK)

# plot
fig, ax = plt.subplots(1, figsize=(6, 6))
ax.plot(sol_ex.t,sol_ex.y.T )
ax.plot(sol_fe[0], sol_fe[1],'-*' )
ax.plot(sol_he[0], sol_he[1],'-o' )
ax.plot(sol_rk.t,sol_rk.y.T, '-d')

ax.autoscale(enable=True, axis='both', tight=True)
ax.set_ylabel(r'$y(t)$')
ax.set_xlabel(r'$t$')
ax.legend(['Exact solution(RK45)','Forward Euler Method',
    'Heuns Method', 'Classical Runge-Kutta'])
plt.tight_layout()     
plt.savefig("euler.png")
