# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import time
start_time=time.time()

# +
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lmfit
from scipy import interpolate
import pickle

from bfunc import bfunc00,bfunc01,bfunc02
# -

# Data load and region parameters

# +
Sample='carina'

samples=pd.read_csv('SampleList//'+Sample+'.csv',header=None)

CarN=dict()
Car=dict()


for i in range(len(samples)):
    CarN[i]=samples[0][i]
    
for i in range(len(samples)):
    Car[i]=pd.read_csv('SFdata//'+CarN[i]+'.csv')    

samples

# +
fig, ax=plt.subplots(figsize=(8,8))

plt.loglog(Car[0].pc,Car[0].S,marker='o',color="blue",alpha=0.65,label="Blue",markersize="8")
plt.loglog(Car[1].pc,Car[1].S,marker='o',color="red",alpha=0.65,label="Red",markersize="8")
plt.loglog(Car[2].pc,Car[2].S,marker='o',color="green",alpha=0.65,label="Comb",markersize="8")

plt.axhline(y=8.1**2, color='blue', linestyle='-')
plt.axvline(x=1.18, color='blue', linestyle='-')
plt.axhline(y=2*8.1**2, color='blue', linestyle=':')

lo=1.18
ss=8.1**2
n=1.5
x=np.linspace(0.25,10,100)
c=1/(1+(x/lo)**n)
y=2*(1-c)*ss
plt.loglog(x,y,color='blue', linestyle='-',linewidth='2.3')

plt.axhline(y=7.1**2, color='red', linestyle='-')
plt.axvline(x=0.57, color='red', linestyle='-')

lo=0.57
ss=7.1**2
n=1.5
x=np.linspace(0.25,10,100)
c=1/(1+(x/lo)**n)
y=2*(1-c)*ss
plt.loglog(x,y,color='red', linestyle='-',linewidth='2.3')


plt.axhline(y=4.22**2, color='green', linestyle='-')
plt.axvline(x=0.95, color='green', linestyle='-')

lo=0.95
ss=5.9**2
n=0.8
x=np.linspace(0.25,10,100)
c=1/(1+(x/lo)**n)
y=2*(1-c)*ss
plt.loglog(x,y,color='green', linestyle='-',linewidth='2.3')

plt.legend()
plt.title('Carina')

ax.text(0.1, 0.9,'m = 1.5', ha='center', va='center', transform=ax.transAxes, color='blue')
ax.text(0.1, 0.85,'m = 1.5', ha='center', va='center', transform=ax.transAxes, color='red')
ax.text(0.1, 0.8,'m = 0.8', ha='center', va='center', transform=ax.transAxes, color='purple')


ax.set(xlabel='r [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)
#plt.grid(which='minor')

# -

B=Car[2].S
r=Car[2].pc

# +
rgrid = np.logspace(0.0, 2)

s0 = (0.011*.92)/2.355             
m = 0.8

sig2 = 4.22**2
r0 = 0.95
# -

model02 = lmfit.Model(bfunc02)
model02.param_names

relative_uncertainty = 0.2
weights = 1.0 / (relative_uncertainty * B)
weights[r > r0] /= 1.5

for p in model02.param_names:
    model02.set_param_hint(p, min=0.0)
model02.set_param_hint("sig2", value=sig2, vary=False)
#model02.set_param_hint("s0", min=0.2)
model02.print_param_hints()

result2 = model02.fit(
    B, 
    weights=weights,
    r=r, r0=r0, m=m, s0=s0, noise=1/10,
)

fig, _ = result2.plot()
fig.axes[0].set(
    xscale="log",
    yscale="symlog",
)
fig.axes[1].set(
    xscale="log",
    yscale="log",
);

print(result2.fit_report())

for p in result2.model.param_names:
    result2.params[p].stderr = result2.params[p].value * 0.1

result2.conf_interval()
print(result2.ci_report())

plt.style.use([
    "seaborn-poster",
])

plot_limits = {
    "s0": [0.0, 0.1],
    "m": [0.1, 1.7],
    "r0": [0.4, 5.0],
    "noise": [0.0, 2.0],
}

# +
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plt.title('M8')
levels = [0.6827, 0.9545, 0.9973]
colors = ["g", "y", "r"]

for ax, [xvar, yvar] in zip(axes.flat, [
    ["s0", "noise"],
    ["r0", "m"],
    ["m", "s0"],
    ["r0", "s0"],
]):
    cx, cy, grid = lmfit.conf_interval2d(
        result2, result2, xvar, yvar, 30, 30,
        limits=[plot_limits[xvar], plot_limits[yvar]],
    )
    ctp = ax.contour(cx, cy, grid, levels, colors=colors)
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)

fig.tight_layout();
# -

x=r
y=B-2*sig2
tck=interpolate.splrep(x,y,s=0)
grid=np.linspace(x.min(),x.max(),num=len(x))
ynew=interpolate.splev(grid,tck,der=0)
inter=pd.DataFrame([grid,ynew]).T
SFr=interpolate.sproot(tck)
SFr

r.max()/result2.params['r0'].value

(r[3]/2**0.5)*((result2.params['r0'].value/r[3])**(result2.params['m'].value/2))

f = open('CI//Car.pkl',"wb")
pickle.dump(result2,f)
f.close()

print("--- %s seconds ---" % (time.time()-start_time))
