import numpy as np
import pylab as plt
from pylab import sqrt
from scipy.integrate import odeint
import matplotlib as mpl
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


def derivate(y, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def second_derivate(y, beta, gamma):
    S, I, R = y
    dS2dt2 = -2 * beta * S * I
    dI2dt2 = 2 * beta * S * I - gamma * I
    dR2dt2 = gamma * I
    return dS2dt2, dI2dt2, dR2dt2


# %%
# Total population, N.
N = 1
days = 150
dt = 0.1
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 0.0001, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.4, 1 / 5
# A grid of time points (in days)
t = np.linspace(0., days, num=int(days / dt) + 1)

# Initial conditions vector
E = np.zeros([len(t), 3])

E[0] = S0, I0, R0


print(beta * S0 / gamma)
# %%
for i in range(1, len(t)):
    # + second_derivate(E[i-1], beta, gamma)[0]*dt*dt*0.5
    E[i, 0] = E[i - 1, 0] + derivate(E[i - 1], beta, gamma)[0] * dt
    # + second_derivate(E[i-1], beta, gamma)[1]*dt*dt*0.5
    E[i, 1] = E[i - 1, 1] + derivate(E[i - 1], beta, gamma)[1] * dt
    # + second_derivate(E[i-1], beta, gamma)[2]*dt*dt*0.5
    E[i, 2] = E[i - 1, 2] + derivate(E[i - 1], beta, gamma)[2] * dt


# %%
# Plot the data on three separate curves for S(t), I(t) and R(t)
X = ret

fig, ax = plt.subplots(figsize=fig_size)
ax.plot(t, E[:, 0], green, label=r"$S$", alpha=1)
ax.plot(t, E[:, 1], red, label=r"$I$", alpha=1)
ax.plot(t, E[:, 2], blue, label=r"$R$", alpha=1)
ax.plot(t, X[:, 0], green, linestyle='-.', alpha=0.4)
ax.plot(t, X[:, 1], red, linestyle='-.', alpha=0.4)
ax.plot(t, X[:, 2], blue, linestyle='-.', alpha=0.4)
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
# for gamma in gammas:
#    ret=odeint(deriv, y0, t, args=(N,beta,gamma,vacc))
#    S, I, R = ret.T
#    maxI=np.amax(I)
#    maxt=np.argmax(I)/100
#    ax.plot(t,I)
#    ax.set_xlabel(r"$Tempo(giorni)$")
#    ax.set_ylabel(r"$Popolazione$")
#    #ax.set_yscale('log')
#    ax.spines['left'].set_position('zero')
#    ax.spines['bottom'].set_position('zero')
#    ax.set_ylim(0,2500)
#    ax.set_xlim(0,days)
#    ax.text(maxt, maxI+30,f"$\\gamma$={gamma:.2f}")#:.2f prime due cifre significative, anche se 0
#
#seaborn.despine(fig, top=True, right='True')
# plt.show()
