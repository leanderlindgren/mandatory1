import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import sqrt
from Wave2D import Wave2D_Neumann

N = 100
sol = Wave2D_Neumann()
xij, yij = sol.create_mesh(N)
data = sol(N = 100, Nt = 100, mx=2, my=2, cfl=1/sqrt(2), store_data=3)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xij, yij, data[0], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# capture, otherwise there will be a plot in this cell
import matplotlib.animation as animation

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
frames = []
for n, val in data.items():
    frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2)
    frames.append([frame])

ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                repeat_delay=1000)
ani.save('report/neumannwave.gif', writer='pillow', fps=8) # This animated png opens in a browser