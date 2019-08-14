import numpy as np
import pylab as plt
from pylab import sqrt
import matplotlib as mpl
import matplotlib.animation as animation
import seaborn

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 2.5 / 72.27               # Convert pt to inch
golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

red = '#E32636'
blue = '#5D8AA8'
green = '#8DB600'

# %%

Moore = [[1, 1], [1, 0], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, 0], [-1, -1]]
vonNeumann = [[1, 0], [0, 1], [0, -1], [-1, 0]]


def p(x):
    r = np.random.rand()
    if r < x:
        return 1
    return 0


def HoodSum(i, j, C):

    for Neighbor in Hood:
        a = i + Neighbor[0]
        b = j + Neighbor[1]
        if a >= size or b >= size or a < 0 or b < 0:
            continue
        if (C[a, b] == 0):
            if (p(beta)):
                return 1
    return 0


def updateMatrices(C):
    nC = np.ones(shape=(size, size))

    for i in range(size):
        for j in range(size):
            if (C[i, j] == 0):
                if (p(gamma)):
                    nC[i, j] = -1
                else:
                    nC[i, j] = 0
            if (C[i, j] == 1):
                if (HoodSum(i, j, C)):
                    nC[i, j] = 0
                else:
                    if (p(vacc)):
                        nC[i, j] = -1
                    else:
                        nC[i, j] = 1
            if (C[i, j] == -1):
                nC[i, j] = -1
    return nC.copy()


# %%
size = 100
Hood = Moore

beta = 0.35
gamma = 0.2
vacc = 0

days = 150

C = np.ones(shape=(size, size))

C[size // 2, size // 2] = 0

listC = []

sScatter = []
iScatter = []
rScatter = []

listC.append(C)

sScatter.append((C == 1).sum())
iScatter.append((C == 0).sum())
rScatter.append((C == -1).sum())

for _ in range(days):
    C = updateMatrices(C)
    sScatter.append((C == 1).sum())
    iScatter.append((C == 0).sum())
    rScatter.append((C == -1).sum())
    listC.append(C)

# %%
t = np.linspace(0, days + 1, days + 1)

fig, ax = plt.subplots(figsize=fig_size)
ax.plot(t, sScatter, green, label=r"$S$")
ax.plot(t, iScatter, red, label=r"$I$")
ax.plot(t, rScatter, blue, label=r"$R$")
ax.set_xlabel(r"$Tempo (giorni)$")
ax.set_ylabel(r"$Popolazione$")
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_ylim(0, size * size)
ax.set_xlim(0, len(t))

legend = ax.legend()
seaborn.despine(fig, top=True, right='True')
plt.show()
plt.savefig(f"SIR_{beta:.2f}_{gamma:.2f}.png")
# %%
fig = plt.figure()
cmap = mpl.colors.ListedColormap([blue, red, green])
ims = [[plt.imshow(matrix, vmin=-1, vmax=1, cmap=cmap)] for matrix in listC]
plt.axis('off')
# plt.xticks([],[])
# plt.yticks([],[])

ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=size,
    repeat_delay=0,
    blit=True)
plt.show()
# %%
timeStamps = [[10, 5], [3, 20],
              [20, 40]]
fig = plt.figure()

for i in range(len(np.ravel(timeStamps))):
    ax = fig.add_subplot(len(timeStamps) * 100 +
                         len(timeStamps[0]) * 10 + (i + 1))
    tempo = np.ravel(timeStamps)[i]
    ax.imshow(listC[tempo], vmin=-1, vmax=1, cmap=cmap)
    ax.text(
        0.95,
        0.01,
        f"t={tempo}",
        verticalalignment='bottom',
        horizontalalignment='right',
        transform=ax.transAxes)
    plt.xticks([], [])
    plt.yticks([], [])

plt.show
