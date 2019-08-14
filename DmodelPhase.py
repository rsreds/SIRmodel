import numpy as np
import pylab as plt
from pylab import sqrt
from scipy.integrate import odeint
import matplotlib as mpl
import matplotlib.ticker as mtick
import seaborn

golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = 5  # width in inches
fig_height = fig_width      # height in inches
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
    S, I = y
    dSdt = -beta * S * I/N + vacc * S
    dIdt = beta * S * I/N - gamma * I
    return dSdt, dIdt

#%%
    
# Total population, N.
N = 1
days=300
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

brs=np.linspace(0.5,5, num=10)
izeros=np.linspace(0.001,0.9, num=10)
gamma, vacc, pVacc = 0.1   ,0.,0.
br=0.3/gamma
# A grid of time points (in days)
t = np.linspace(0., days, num=(days))
I0,R0,S0=10,0,N-10
izero=10/2500
color=[red, green, blue,red, green, blue,red, green, blue,red, green, blue,red, green, blue,red, green, blue,red, green, blue,red, green, blue]
S,I=[],[]
index=0
fig, ax = plt.subplots()
x=np.linspace(0,1,10)
y=np.linspace(1,0,10)

ax.plot(x,y,color='black', linewidth=0.5)
for br in brs:
        index=index+1       
        I0, R0 = izero*N, (pVacc/100)*N
        S0 = N - I0 - R0
        y0 = S0, I0
        ret = odeint(deriv, y0, t, args=(N, br*gamma, gamma, vacc))
        S,I=ret.T
        indmaxI=np.argmax(I)
        SImax=S[indmaxI]
        print(f"{SImax:.3f}\t{br:.3f}\t{1/br:.3f}")
        ax.quiver(ret[:-1,0],ret[:-1,1],ret[1:,0]-ret[:-1,0],ret[1:,1]-ret[:-1,1],scale_units='xy', angles='xy', scale=1.05, color=color[index], width=0.003)

ax.set_xlabel(r"$Suscettibili$")
ax.set_ylabel(r"$Infetti$")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
#plt.plot(S,I,blue)
   
plt.show
