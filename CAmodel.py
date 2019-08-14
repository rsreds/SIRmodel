import numpy as np
import pylab as plt
from pylab import sqrt
import matplotlib.animation as animation
import seaborn

golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
fig_width = 3.6  # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 12,
          'font.size': 12,
          'figure.autolayout': True,
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
CAside = 50
time = 150

t = np.linspace(0, time + 1, time + 1)

N = CAside * CAside
CAshape = (CAside, CAside)
S = np.ones(shape=CAshape)
I = np.zeros(shape=CAshape)
R = np.zeros(shape=CAshape)


I[CAside // 2, CAside // 2] = 0.1
S[CAside // 2, CAside // 2] = 1-I[CAside // 2, CAside // 2]

Moore = [[1, 1], [1, 0], [1, -1], [0, 1],
         [0, -1], [-1, 1], [-1, 0], [-1, -1], [0, 0]]
VNeumann = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 0]]
MeanField = [[0, 0]]
Cebyshev = [[0, 0]]
All = [[0, 0]]

Hood = MeanField

beta = 0.4
gamma = 0.2
nu = 1./N
vacc = 0
deterministic = True


listS = []
sScatter = []
listS.append(S)
sScatter.append(np.floor(np.sum(S) * 100) / (N * 100))

listI = []
iScatter = []
listI.append(I)
iScatter.append(np.floor(np.sum(I) * 100) / (N * 100))
listR = []
rScatter = []
listR.append(R)
rScatter.append(np.floor(np.sum(R) * 100) / (N * 100))

# %%


def p(x):
    if deterministic:
        return x
    else:
        sigma = 0.3
        r = np.random.randn()
        result = sigma * r + x
        if result <= 0:
            return 0
        if result >= 1:
            return 1
        return sigma * r + x


def HoodSum(i, j):
    global I
    result = 0
    
    if (Hood==All):
        for a in range (CAside):
            for b in range(CAside):
                result = result + p(nu) * I[a, b]
                
    if (Hood==Cebyshev):
        for u in range(-3,4):
            for v in range(-3,4):
                a = i + u
                b = j + v
                if a >= CAside or b >= CAside or a < 0 or b < 0:
                    continue
                result = result + (1/(np.maximum(np.abs(u),np.abs(v))+1))*p(nu) * I[a, b]
        return result
        
    for Neighbor in Hood:
        a = i + Neighbor[0]
        b = j + Neighbor[1]
        if a >= CAside or b >= CAside or a < 0 or b < 0:
            continue
        result = result + p(nu) * I[a, b]
    if (Hood == MeanField):
        for _ in range(3):
            a = np.random.randint(0, CAside)
            b = np.random.randint(0, CAside)
            result = result + nu * I[a, b]

    return result


def updateMatrices():
    global S
    global I
    global R
    nS = np.ones(shape=CAshape)
    nI = np.zeros(shape=CAshape)
    nR = np.zeros(shape=CAshape)

    for i in range(CAside):
        for j in range(CAside):
            g = p(gamma)
            nR[i, j] = R[i, j] + g * I[i, j]

            if nR[i, j] > 1.:
                nR[i, j] = 1.

            nI[i, j] = (1. - g) * I[i, j] + p(beta) * S[i, j] * HoodSum(i, j)

            if nI[i, j] > 1. - nR[i, j]:
                nI[i, j] = 1. - nR[i, j]

            nS[i, j] = 1. - nR[i, j] - nI[i, j]
#    if(u==25):
#        nS[40,40]=0
#        nI[40,40]=1
#        nR[40,40]=0
    S = nS.copy()
    I = nI.copy()
    R = nR.copy()
# %%


for u in range(time):
    updateMatrices()
    sScatter.append(np.floor(np.sum(S) * 100) / (N * 100))
    iScatter.append(np.floor(np.sum(I) * 100) / (N * 100))
    rScatter.append(np.floor(np.sum(R) * 100) / (N * 100))
    listS.append(S)
    listI.append(I)
    listR.append(R)
    
print(rScatter[len(rScatter)-1])
# %%

fig, ax = plt.subplots(figsize=fig_size)
ax.plot(t, sScatter, green, label=r"$S$")
ax.plot(t, iScatter, red, label=r"$I$")
ax.plot(t, rScatter, blue, label=r"$R$")
ax.set_xlabel(f"$Tempo\\quad t$")
ax.set_ylabel(f"$Fraz.\\;di\\;popol.$")
# ax.set_yscale('log')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_ylim(0, 1)
ax.set_xlim(0, time)

legend = ax.legend()
seaborn.despine(fig, top=True, right='True')
plt.show()

# %%
# Animated plot
fig = plt.figure()
plt.subplot(1, 2, 1)
SIRplot = plt.plot(rScatter, blue, sScatter, green, iScatter, red)
plt.subplot(1, 2, 2)
ims = [[plt.imshow(matrix, vmin=0, vmax=0.2, cmap='OrRd')] for matrix in listI]
ani = animation.ArtistAnimation(
    fig,
    ims,
    interval=CAside,
    repeat_delay=0,
    blit=True)
plt.show()
# %%
# Stack plot
params = {'backend': 'ps',
          'axes.labelsize': 24,
          'font.size': 24,
          'figure.autolayout': True,
          'legend.fontsize': 24,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

timeStamps = [[0,5],[10,25],[50,75]]
fig_width = 5
fig_size = [fig_width, fig_width*1.5]
fig = plt.figure(figsize=fig_size)

pos = np.linspace(0, CAside, CAside)


for i in range(len(np.ravel(timeStamps))):
    ax = fig.add_subplot(len(timeStamps) * 100 +
                         len(timeStamps[0]) * 10 + (i + 1))
    tempo = np.ravel(timeStamps)[i]
    ax.imshow(listI[tempo], vmin=0, vmax=max(iScatter), cmap='OrRd')
    ax.text(
        0.95,
        0.01,
        f"$t={tempo}$",
        verticalalignment='bottom',
        horizontalalignment='right',
        transform=ax.transAxes)
    ax.grid(color=blue, linestyle='-', linewidth=0.2)
    ax.tick_params(direction='out', length=6, width=0, colors='r')
    plt.xticks(pos, [])
    plt.yticks(pos, [])


