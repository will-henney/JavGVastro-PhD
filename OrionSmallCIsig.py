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
data = json.load(open("SFdata//OrionSmall.json"))

pixscale = 0.534 # arcsec
pixscale *= 0.00242               # parsec
s0 = 0.00242/2.355              # parsec
m = 1.5

# +
r = pixscale * 10**np.array(data["log10 r"])
rgrid = pixscale * np.logspace(0.0, 2.5)

B = np.array(data["Unweighted B(r)"])

sig2 = data["Unweighted sigma^2"]
#r0 = np.interp(sig2, B, r)
r0=0.06
# -

B

B=B[9:55]
r=r[9:55]

# +
fig, ax = plt.subplots(figsize=(8, 8))
# Plot fit to unweighted strucfunc
ax.plot(rgrid, bfunc02(rgrid, r0, sig2, m, s0, 1/10), color="red")
ax.plot(rgrid, bfunc00(rgrid, r0, sig2, m), color="0.8")
# Plot points from unweighted strucfunc
ax.plot(r, B, 'o',  color='black')

ax.axhline(sig2)
ax.axvline( 2.355  * s0, linestyle="dashed")
ax.axvline(r0, linestyle="dotted")

ax.set(
    xscale = "log",
    yscale = "log",
#    ylim  = [1, 250],
#    xlim  = [1, 150],
    xlabel = "r [pc]",
    ylabel = r"B(r) [km$^{2}$/s$^{2}$]",
)

sig2,r0
# -

model02 = lmfit.Model(bfunc02)
model02.param_names

relative_uncertainty = 0.15
weights = 1.0 / (relative_uncertainty * B)
weights[r > r0] /= 3

for p in model02.param_names:
    model02.set_param_hint(p, min=0.0)
#model02.set_param_hint("sig2", value=sig2, vary=False)
model02.print_param_hints()

result2 = model02.fit(
    B, 
    weights=weights,
    r=r, r0=r0, m=m, s0=s0, noise=1/10, sig2=sig2
)

# +
fig, _ = result2.plot( 'ko')
fig.axes[0].set(
    title='Orion Small',
    xscale="log",
    yscale="symlog",
)
fig.axes[1].set(
    xscale = "log",
    yscale = "log",
    xlabel = "r [pc]",
    ylabel = r"B(r) [km$^{2}$/s$^{2}$]",
);

plt.savefig('SFpaper//OS.pdf', bbox_inches='tight')


# -

print(result2.fit_report())

# +
#for p in result2.model.param_names:
#    result2.params[p].stderr = result2.params[p].value * 0.1
# -

result2.conf_interval()
print(result2.ci_report())

plt.style.use([
    "seaborn-poster",
])

plot_limits = {
    "s0": [0.0001, 0.006],
    "m": [1.1, 1.8],
    "r0": [0.04, 0.1],
    "noise": [0.0, 0.5],
}

# +
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plt.title('Orion S')
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

(r[0]/2**0.5)*((result2.params['r0'].value/r[0])**(result2.params['m'].value/2))

f = open('CI//OrionS.pkl',"wb")
pickle.dump(result2,f)
f.close()





print("--- %s seconds ---" % (time.time()-start_time))
