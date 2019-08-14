import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


u0, v0 = 0.00009, 0.0001
a1,a2,b,g = 10,10,4,4

# A grid of time points (in days)
t = np.linspace(0, 10, 100)


def deriv(y, t, a1,a2,b,g):
    u, v = y
    dUdt = a1/(1+v**b)-u
    dVdt = a2/(1+u**g)-v
    
    return dUdt, dVdt

# Initial conditions vector
y0 = u0,v0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(a1,a2,b,g))
U,V = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
ax.plot(t, U, 'g', alpha=0.5, lw=2, label='u')
ax.plot(t, V, 'r', alpha=0.5, lw=2, label='v')

ax.set_xlabel('Time ')
ax.set_ylabel('concentration')
ax.set_ylim(0,10)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
