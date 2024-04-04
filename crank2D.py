import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


# initialbetingelse
def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# parametre
T = 500
X = 10
Y = 10
time = 0.1
L = 1
hx = L / X
hy = L / Y
k = time / T

# løsningsmatrise
u = np.zeros((X + 1, Y + 1, T + 1))

# initialbetingelser
x = np.linspace(0, 1, X + 1)
y = np.linspace(0, 1, Y + 1)
u[:, :, 0] = f(
    x[:, None], y[None, :]
)  # [:,None] og [None,:] gjør at vi får en matrise med x-verdier i kolonner og y-verdier i rader

print(x)
print(x[:, None])

# gamma
gamma1 = k / (hx**2)
gamma2 = k / (hy**2)
print(gamma1)
print(gamma2)


# matrise A
A = np.zeros((X + 1, Y + 1, X + 1, Y + 1))
A[0, :, 0, :] = 1
A[X, :, X, :] = 1
A[:, 0, :, 0] = 1
A[:, Y, :, Y] = 1

# matrise B
B = np.zeros((X + 1, Y + 1, X + 1, Y + 1))
B[0, :, 0, :] = 1
B[X, :, X, :] = 1
B[:, 0, :, 0] = 1
B[:, Y, :, Y] = 1

for x in range(1, X):
    for y in range(1, Y):
        A[x, y, x - 1, y] = -gamma1
        A[x, y, x, y - 1] = -gamma2
        A[x, y, x, y] = 1 + 2 * gamma1 + 2 * gamma2
        A[x, y, x + 1, y] = -gamma1
        A[x, y, x, y + 1] = -gamma2

for x in range(1, X):
    for y in range(1, Y):
        B[x, y, x - 1, y] = gamma1
        B[x, y, x, y - 1] = gamma2
        B[x, y, x, y] = 1 - 2 * gamma1 - 2 * gamma2
        B[x, y, x + 1, y] = gamma1
        B[x, y, x, y + 1] = gamma2


"""
Matrisene A og B er 4D-arrays, hvor de to første dimensjonene er indeksene til punktet i rutenettet, og de to siste dimensjonene er indeksene til nabopunktene.
Dette er for å kunne reshape matrisene til 2D-arrays som gir mening, slik at vi kan bruke np.linalg.pinv og np.dot.
Disse er ikke kvadratiske matriser, så vi kan ikke bruke np.linalg.solve. Vi må derfor bruke np.linalg.pinv, som gir pseudoinversen til en matrise.
Bruker ellers samme løsningsmetode som man ville gjort om man løste et helt vanlig likningssett på formen Ax = Bb. hvor A og B er matriser og x er ukjent vektor og b er kjent vektor.
"""


# løs varmeligning
for t in tqdm(range(T)):
    u[:, :, t + 1] = (
        np.linalg.pinv(A.reshape((X + 1) * (Y + 1), (X + 1) * (Y + 1)))
        .dot(B.reshape((X + 1) * (Y + 1), (X + 1) * (Y + 1)))
        .dot(u[:, :, t].reshape((X + 1) * (Y + 1)))
        .reshape(X + 1, Y + 1)
    )


# print(u[5,5,:])

# plot løsning
x = np.linspace(0, 1, X + 1)
y = np.linspace(0, 1, Y + 1)

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection="3d")


def update(frame, plot, u, X, Y):
    ax.clear()  
    plot = ax.plot_surface(X, Y, u[:, :, frame], cmap="viridis", edgecolor="none", vmin=0, vmax=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title(f"Varmeligningen for t =  {frame*k:.2f}")

    return (plot,)


plot = ax.plot_surface(X, Y, u[:, :, 0], cmap="viridis", edgecolor="none")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_title("Varmeligningen for t = 0")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
cbar = fig.colorbar(plot)

animation = FuncAnimation(
    fig, update, frames=T + 1, fargs=(plot, u, X, Y), interval=1, blit=False
)
plt.show()

animation.save('crank2D.gif', writer='pillow')  # lagrer animasjonen som en gif





