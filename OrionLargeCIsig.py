# -*- coding: utf-8 -*-
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
Sample='orion'

samples=pd.read_csv('SampleList//'+Sample+'.csv',header=None)

DataNO=dict()
DataO=dict()


for i in range(len(samples)):
    DataNO[i]=samples[0][i]
    
for i in range(len(samples)):
    DataO[i]=pd.read_csv('SFdata//'+DataNO[i]+'.csv')    



samples
# -



# +
fig, ax=plt.subplots(figsize=(8,8))

plt.loglog(DataO[5].pc,DataO[5].S,color="black",alpha=0.65)

plt.axhline(y=3.23**2, color='black', linestyle='-')
plt.axvline(x=0.55, color='black', linestyle='-')

lo=0.55
ss=3.23**2
n=0.75
x=np.linspace(0.1,3,100)
c=1/(1+(x/lo)**n)
y=2*(1-c)*ss
plt.loglog(x,y,color='r', linestyle='-',linewidth='2.3')

#ax.text(0.83, 0.15,'m = 1.15', ha='center', va='center', transform=ax.transAxes, color='red')
#ax.text(0.83, 0.20,'r$_{0}$ = 1.5 pc', ha='center', va='center', transform=ax.transAxes, color='red')
#ax.text(0.84, 0.25,'σ = 2.7 km/s', ha='center', va='center', transform=ax.transAxes, color='red')
    
#ax.set(xlabel='r [pc]', ylabel='B(r) [km$^{2}$/s$^{2}$]')
plt.tick_params(which='both', labelright=False, direction='in', right=True,  top=True)



# +
rgrid = np.logspace(0.1, 3)

s0 = (0.00242)/2.355             
m = 0.75

sig2 = ss
r0 = 0.55

B=DataO[5].S[0:6]
r=DataO[5].pc[0:6]
# -

model02 = lmfit.Model(bfunc02)
model02.param_names

relative_uncertainty = 0.1
weights = 1.0 / (relative_uncertainty * B)
weights[r > r0] /= 1.25

# +
for p in model02.param_names:
    model02.set_param_hint(p, min=0.0)

#model02.set_param_hint("sig2", value=sig2, vary=False)
#model02.set_param_hint("s0", min=0.2)
model02.print_param_hints()
# -

B

result2 = model02.fit(
    B, 
    weights=weights,
    r=r, r0=r0, m=m, s0=s0, noise=1/10, sig2=sig2
)

# +
fig, _ = result2.plot()
fig.axes[0].set(
    xscale="log",
    yscale="symlog",
)
fig.axes[1].set(
    xscale="log",
    yscale="log",
);


# -

print(result2.fit_report())

for p in result2.model.param_names:
    result2.params[p].stderr = result2.params[p].value * 0.1

result2.conf_interval()
print(result2.ci_report())

plt.style.use([
    "seaborn-poster",
])

plot_limits = {
    "s0": [0.0, 1],
    "m": [0.0, 3.0],
    "r0": [0.0, 8.0],
    "noise": [0.0, 2.0],
}

# +
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plt.title('Orion L')
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

r.max()

r.max()/result2.params['r0'].value

(r[0]/2**0.5)*((result2.params['r0'].value/r[0])**(result2.params['m'].value/2))

f = open('CI//OrionL.pkl',"wb")
pickle.dump(result2,f)
f.close()



print("--- %s seconds ---" % (time.time()-start_time))
