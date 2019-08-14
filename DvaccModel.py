import numpy as np
import pylab as plt
from pylab import sqrt
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.ticker as mtick
import seaborn

golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = 9  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

params = {'backend': 'ps',
          'axes.labelsize': 12,
          'font.size': 12,
          'legend.fontsize': 12,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'text.usetex': True,
          'figure.figsize': fig_size}
plt.rcParams.update(params)

red='#E32636'
blue='#5D8AA8'
green='#8DB600'

#%%
def deriv(y, t, N, beta, gamma, vacc):
    S, I, R = y
    dSdt = -beta * S * I/N - vacc * S
    dIdt = beta * S * I/N - gamma * I
    dRdt = gamma * I + vacc * S
    return dSdt, dIdt, dRdt

#%%
    
# Total population, N.
N = 2500
days=15
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
betas=np.linspace(0.2,2,num=10)
gammas=np.linspace(0.05,0.5, num=10)
beta, gamma, vacc = 1, 0.1,0.
# A grid of time points (in days)
t = np.linspace(0., days, num=days)
I0,R0,S0=10,0,N
brs=[]
trs=[]
for b in betas:
    for g in gammas:
        for i in range(101):
            pVacc=i
            beta,gamma=b,g
            br=beta/gamma
            I0, R0 = 10, (pVacc/100)*N
            S0 = N - I0 - R0
            y0 = S0, I0, R0
            ret = odeint(deriv, y0, t, args=(N, beta, gamma, vacc))
            S, I, R = ret.T
            if (I[2]<I[1]):
                brs.append(br)
                trs.append(i)
                
                break
fig, ax = plt.subplots(figsize=fig_size)
ax.plot(brs,trs,'.',color=red)
ax.set_xlabel(r"$R_0$")
ax.set_ylabel(r"$Vaccinati$")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))


    
plt.show
#%%
# Plot the data on three separate curves for S(t), I(t) and R(t)

fig,ax=plt.subplots(figsize=fig_size)
ax.plot(t,S,green,label=r"$S$")
ax.plot(t,I,red,label=r"$I$")
ax.plot(t,R,blue,label=r"$R$")
ax.set_xlabel(r"$Tempo (giorni)$")
ax.set_ylabel(r"$Popolazione$")
#ax.set_yscale('log')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_ylim(0,2500)
ax.set_xlim(0,days)

legend = ax.legend()
seaborn.despine(fig, top=True, right='True')
plt.show()
plt.savefig(f"SIR_{beta:.2f}_{gamma:.2f}.png")
#fig = plt.figure(facecolor='w')
#ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
#ax.plot(t, S, 'g', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
#ax.plot(t, R, 'b', alpha=0.5, lw=2, label='Removed')
#ax.set_xlabel('Time /days')
#ax.set_ylabel('Number (1000s)')
#ax.set_ylim(0,2500)
#ax.yaxis.set_tick_params(length=0)
#ax.xaxis.set_tick_params(length=0)
#ax.grid(b=True, which='major', c='w', lw=2, ls='-')
#legend = ax.legend()
#legend.get_frame().set_alpha(0.5)
#for spine in ('top', 'right', 'bottom', 'left'):
#    ax.spines[spine].set_visible(False)
#plt.show()
