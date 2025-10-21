import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x = np.linspace(0, 10)
y = np.sin(x)

fig, ax = plt.subplots()
line, = ax.plot(x, y)

def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    return line,

ani = animation.FuncAnimation(fig, update, len(x), interval=100, 
                              fargs=[x, y, line], blit=True)
ani.save('animation_drawing.gif', writer='imagemagick', fps=60)