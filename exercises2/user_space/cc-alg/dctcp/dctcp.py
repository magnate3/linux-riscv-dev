#!/usr/bin/python

import sys
import numpy as np
import matplotlib.pyplot as plt
import random

type = "0"
beta = 0.5

def dxdt(x, y, t):
    if y > 0:
        return - 0.5 * y * x
    else:
        return 1

def dzdt(z, y, t):
    if y > 0:
        return - beta * z
    else:
        return 1

def ydt(y, t):
    if type == "sin":
        return 0.5*(-np.cos(0.1*t) +  np.cos(0.4*t))
    elif type == "xishu":
        prob = random.random()
        if prob < 0.8:
            return 0
        return 2*random.random() - 1
    elif type == "miji":
        return 2*random.random() - 1
    return 0

if len(sys.argv) < 3:
    sys.exit()

type = sys.argv[1]
beta = float(sys.argv[2])
png_name = "cc-{}-{}.png".format(type,beta)
t = np.linspace(0, 50, 500)
x = np.zeros_like(t)
y = np.zeros_like(t)
z = np.zeros_like(t)

x[0], z[0] = 5, 5

for i in range(1, len(t)):
    dt = t[i] - t[i - 1]
    dy = ydt(y[i - 1], i)
    dx = dxdt(x[i - 1], dy, t[i - 1])
    dz = dzdt(z[i - 1], dy, t[i - 1])
    x[i] = x[i - 1] + (dx) * dt
    z[i] = z[i - 1] + (dz) * dt
    y[i] = dy

plt.plot(t, y, label='y') # mark rate < 1
plt.plot(t, x, label='x') # dctcp
plt.plot(t, z, label='z') # reno/cubic
plt.legend()
plt.xlabel('t')
plt.grid(True)
plt.savefig(png_name)
#plt.show()
