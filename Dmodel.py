import numpy as np
import pylab as plt
from pylab import sqrt
from scipy.integrate import odeint
import seaborn

golden_mean = (sqrt(5) - 1.0) / 2.0         # Aesthetic ratio
fig_width = 3.6  # width in inches
fig_height = fig_width * golden_mean      # height in inches
fig_size = [fig_width, fig_height]

params = {'backend': 'ps',
          'axes.labelsize': 11,
          'font.size': 10,
          'figure.autolayout': True,
          'legend.fontsize': 9,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'text.usetex': True,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

red = '#E32636'
blue = '#5D8AA8'
green = '#8DB600'

# %%


def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# %%
# Total population, N.
N = 1
days = 150
dt = 1
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 0.0001, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.4, 0.2
betas = np.linspace(0.35, 1.4, 4)
gammas = np.linspace(1 / 20, 1 / 5, 4)

# A grid of time points (in days)
t = np.linspace(0., days, num=int(days / dt) + 1)

# Initial conditions vector
y0 = S0, I0, R0

print(beta * S0 / gamma)
# %%
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

print(R[len(R)-1])
# %%
# Plot the data on three separate curves for S(t), I(t) and R(t)

fig, ax = plt.subplots(figsize=fig_size)
ax.plot(t, S, green, label=r"$S$")
ax.plot(t, I, red, label=r"$I$")
ax.plot(t, R, blue, label=r"$R$")
ax.set_xlabel(f"$Tempo\\quad t$")
ax.set_ylabel(f"$Fraz.\\;di\\;popol.$")
# ax.set_yscale('log')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_ylim(0, 1)
ax.set_xlim(0, days)

legend = ax.legend()
seaborn.despine(fig, top=True, right='True')
plt.show()
# plt.savefig(f"SIR_{beta:.2f}_{gamma:.2f}_{pVacc*100}%.png")

# %%

# fig,ax=plt.subplots(figsize=fig_size)
#
# for beta in betas:
#    ret=odeint(deriv, y0, t, args=(N,beta,gamma))
#    S, I, R = ret.T
#    maxI=np.amax(I)
#    maxt=np.argmax(I)
#    ax.plot(t,I,label=f"$\\beta$={beta:.2f}")
#    ax.set_xlabel(r"$Tempo(giorni)$")
#    ax.set_ylabel(r"$Popolazione$")
#    #ax.set_yscale('log')
#    ax.spines['left'].set_position('zero')
#    ax.spines['bottom'].set_position('zero')
#    ax.set_ylim(0,1)
#    ax.set_xlim(0,days)
# ax.text(maxt+3, maxI+0.005,f"$\\gamma$={gamma:.2f}")
#    legend=ax.legend()
#seaborn.despine(fig, top=True, right='True')
# plt.show()
