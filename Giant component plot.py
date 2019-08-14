import numpy as np
import pylab as plt
from pylab import sqrt
from scipy.integrate import odeint
import matplotlib as mpl
import seaborn

golden_mean = (sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = 3.6 # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

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

red='#E32636'
blue='#5D8AA8'
green='#8DB600'
#%%
bgs=2000
bg=np.linspace(0, 5, bgs)
R=np.linspace(0,1,1000)
y=np.empty([bgs,1000])
z=np.empty([3,1000])
s=np.empty([bgs])
R=np.round(R,3)
for i in range(bgs):
    for j in range(1000):    
        y[i,j]=np.round(1-np.exp(-bg[i]*R[j]),3)
for j in range(1000):    
        z[0,j]=1-np.exp(-0.5*R[j])
        z[1,j]=1-np.exp(-1*R[j])
        z[2,j]=1-np.exp(-1.5*R[j])
for i in range(bgs):
    for j in range(1000):    
        if (y[i,j]==R[j]):
            s[i]=R[j]

#%%
# Plot the data on three separate curves for S(t), I(t) and R(t)
fig,ax=plt.subplots(figsize=[fig_width,fig_width])

ax.plot(R,z[0],color=blue)
ax.plot(R,z[1],color=blue)
ax.plot(R,z[2],color=blue)
ax.plot(R,R,'--',color=red)

ax.text(0.6,0.2,f'$\\beta/\\gamma=0.5$')

ax.text(0.6,0.4,f'$\\beta/\\gamma=1$')

ax.text(0.2,0.5,f'$\\beta/\\gamma=1.5$')

ax.set_xlabel(f"$R$")
ax.set_ylabel(f"$y$")
#ax.set_yscale('log')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_ylim(0,1)
ax.set_xlim(0,1)

seaborn.despine(fig, top=True, right='True')

plt.show()
#plt.savefig(f"SIR_{beta:.2f}_{gamma:.2f}_{pVacc*100}%.png")

#%%
fig,ax=plt.subplots(figsize=fig_size)

ax.plot(bg,s,color=blue)


ax.set_xlabel(f"$\\beta/\\gamma$")
ax.set_ylabel(f"$R$")

#a2x.set_yscale('log')
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_ylim(0,1)
ax.set_xlim(0,5)
seaborn.despine(fig, top=True, right='True')

plt.show()
