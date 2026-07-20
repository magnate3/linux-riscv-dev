import numpy as np
import matplotlib.pyplot as plt

x0, y0, z0 = 0.1, 5, 2
T, dt = 10, 0.1
k1, k3 = 2, 2
a = 1.25
times = np.arange(0, T + dt, dt)

plt.figure(figsize = (20, 10))

x = np.zeros_like(times)
y = np.zeros_like(times)
z = np.zeros_like(times)
w = np.zeros_like(times)

x[0], y[0], z[0] = x0, y0, z0
for n in range(1, len(times)):
    x[n] = x[n-1] + dt * (-k1*(z[n-1] - y[n-1]))
    y[n] = y[n-1] + dt * (-k3*(y[n-1] - z[n-1]))
    z[n] = 1 + 1*np.sin(2 * np.pi * times[n-1])
    w[n] = x[n]*z[n]
plt.plot(times, x, label = 'x(t)', color='black')
plt.plot(times, y, label = 'y(t)', color='yellow', linestyle = 'dashed')    
plt.plot(times, z, label = 'z(t)', color='red', linestyle = 'dotted')    
plt.plot(times, w, label = 'w(t)', color='orange', linestyle = 'dashdot')    

plt.title(f'k1=2,k2=2')
plt.xlabel('time (t)')
plt.ylabel('vegas')
plt.grid(True)

#plt.show()
plt.legend()
plt.tight_layout()     
plt.savefig("vegs.png")
