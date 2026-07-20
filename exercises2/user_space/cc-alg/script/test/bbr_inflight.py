import numpy as np
import matplotlib.pyplot as plt

x0, y0, z0 = 0.1, 5, 2
T, dt = 200, 0.1
C, R = 10, 2
a = 1.25
times = np.arange(0, T + dt, dt)

plt.figure(figsize = (20, 10))

x = np.zeros_like(times)
y = np.zeros_like(times)
z = np.zeros_like(times)

x[0], y[0], z[0] = x0, y0, z0
for n in range(1, len(times)):
    x[n] = x[n-1] + dt * (C*R*a*x[n-1]/(a*x[n-1] + y[n-1] + z[n-1]) - x[n-1])
    y[n] = y[n-1] + dt * (C*R*a*y[n-1]/(a*y[n-1] + x[n-1] + z[n-1]) - y[n-1])
    z[n] = z[n-1] + dt * (C*R*a*z[n-1]/(a*z[n-1] + y[n-1] + x[n-1]) - z[n-1])

plt.plot(times, x, label = 'x(t)')
plt.plot(times, y, label = 'y(t)', linestyle = 'dashed')    
plt.plot(times, z, label = 'z(t)', linestyle = 'dotted')    

plt.title(f'g = 1.25, C=10, R=2')
plt.xlabel('time (t)')
plt.ylabel('inflight')
plt.grid(True)

#plt.show()
plt.tight_layout()     
plt.savefig("inflight.png")
