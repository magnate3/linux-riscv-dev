import numpy as np
import matplotlib.pyplot as plt

x0, y0 = 0.1, 0.1
T, dt = 50, 0.1
a,b,c = 1, 0.5, 0.1
times = np.arange(0, T + dt, dt)

plt.figure(figsize = (20, 10))

x = np.zeros_like(times)
y = np.zeros_like(times)

x[0], y[0], = x0, y0
for n in range(1, len(times)):
    x[n] = x[n-1] + dt * ((1-c)*a/x[n-1] - c*b*x[n-1])
    y[n] = y[n-1] + dt * ((1-2*c)*a/y[n-1] - 2*c*b*y[n-1])

plt.plot(times, x, label = 'x(t)')
plt.plot(times, y, label = 'y(t)', linestyle = 'dashed')    

plt.title(f'a = 1.0, b=0.5, c=0.1')
plt.xlabel('drop')
plt.ylabel('cwin size')
plt.grid(True)

#plt.show()
plt.tight_layout()     
plt.savefig("aimd.png")
