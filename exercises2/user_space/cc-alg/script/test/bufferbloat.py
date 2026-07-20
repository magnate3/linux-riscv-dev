import numpy as np
import matplotlib.pyplot as plt
beta = 0.5
C, R = 50, 2

# x, z 分别为两条流的 cwnd
def dxdt(x, y, t):
    if y > 0:
        return - beta * x
    else:
        return 1

def dzdt(z, y, t):
    if y > 0:
        return - beta * z
    else:
        return 1

def ydt(a, b, t):
    buff = 2 * C * R
    if (a + b) > buff:
        return 1
    elif False and (a + b) > (buff - int(buff / 4)):
        ret = 2*random.random() - 1
        return ret
    return 0

t = np.linspace(0, 800, 8000)
x = np.zeros_like(t)
y = np.zeros_like(t)
z = np.zeros_like(t)

x[0], z[0] = 1, 250

for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    dxy = ydt(x[i - 1], z[i - 1], i)
    dzy = ydt(z[i - 1], x[i - 1], i)
    dx = dxdt(x[i - 1], dxy, t[i - 1])
    dz = dzdt(z[i - 1], dzy, t[i - 1])
    x[i] = x[i - 1] + (dx) * dt
    z[i] = z[i - 1] + (dz) * dt
plt.plot(t, x, label = 'x(t)-aimd', color='black')
plt.plot(t, z, label = 'z(t)-bbr', color='red', linestyle = 'dotted')    

plt.title(f'buffer = C*R  RED=0  C=50 R=2 ')
plt.xlabel('time (t)')
plt.ylabel('infight')
plt.grid(True)

#plt.show()
plt.legend()
plt.tight_layout()     
plt.savefig("bufferboat.png")
