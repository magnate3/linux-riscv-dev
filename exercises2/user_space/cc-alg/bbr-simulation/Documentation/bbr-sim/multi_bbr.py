#!/opt/homebrew/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.ticker import MultipleLocator

C, R, g, phi, period = 10.0, 1, 1.25, 7, 8
# 定义周期函数
def periodic_function(t, period):
    step = 0.25 / (period / 2)
    position = t % period
    if position < period / 2:
        return 1 + position * step
    else:
        return 1.25 - (position - period / 2) * step

def custom_periodic_function(x, period = 8):
    if int(x) % period  <= 1:
        return g
    elif 1 < int(x) % period <= period:
        return 1
    else:
        return 1

def bbr_model(variables, t):
    x, y, x_i, y_i = variables
    g_x = custom_periodic_function(t, period = 8)
    g_y = custom_periodic_function(t + phi, period = 8)

    dxdt = C * (g_x * x / (g_x * x + g_y * y)) - x
    dydt = C * (g_y * y / (g_y * y + g_x * x)) - y
    dx_idt = x * R - x_i
    dy_idt = y * R - y_i
    return [dxdt, dydt, dx_idt, dy_idt]

initial_conditions = [2.0, 8.0, 2.0, 8.0]

t = np.linspace(0, 350, 22000)

solution = odeint(bbr_model, initial_conditions, t)

x = solution[:, 0]
y = solution[:, 1]
x_i = solution[:, 2]
y_i = solution[:, 3]

r = (x_i + y_i) / C
alpha = 0.02
r_ema = [r[0]]
for i in range(1, len(r)):
    r_ema.append(alpha * r[i] + (1 - alpha) * r_ema[i - 1])

fig, a = plt.subplots(3, 1)
a[0].plot(t, x, label='x')
a[0].plot(t, y, label='y')
a[0].set_ylabel('bandwidth')
a[0].set_title(f'gain = {g}, phi = {phi}, period = {period}, C = {C}, R = {R}')
a[0].legend()

a[1].plot(t, x_i, label='x_i')
a[1].plot(t, y_i, label='y_i')
a[1].set_xlabel('time')
a[1].set_ylabel('inflt')
a[1].legend()

a[2].plot(t, r, label='rtt')
a[2].plot(t, r_ema, label='srtt')
a[2].yaxis.set_major_locator(MultipleLocator(0.5))
a[2].set_xlabel('time')
a[2].set_ylabel('RTT')
a[2].legend()

plt.show()
