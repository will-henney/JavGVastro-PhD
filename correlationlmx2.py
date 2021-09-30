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

# + [markdown] id="K7jaxBz1-k7o"
# This is Will's edit of a notebook originally written by Javier.
#
# # Correlations between H II region parameters
#
# We look at correlations between 6 principal measurements that fall into two groups: 
#
# * Basic parameters: 
#     * Size: $L$
#     * Ionizing luminosity: $Q(\mathrm{H})$
#     * Distance: $D$
# * Velocity structure function parameters:
#     * Velocity dispersion on plane of sky: $\sigma$
#     * Velocity autocorrelation length scale: $\ell_0$
#     * Structure function slope in inertial range: $m$
#
# Colab's markdown renderer seems to have a bug that requires some math outside of a list in order to trigger latex parsing: $\alpha$. 

# + [markdown] id="aZa8qfxAAshr"
# ## Original table from Javier

# + id="y_1iISme7_9P"
import time
start_time=time.time()
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import lmfit
import statsmodels.api as sm
import linmix
from scipy.stats import pearsonr
#import sys
# -

Region =["NGC 604","NGC 595","Hubble X","Hubble V","30Dor" ,"Carina","NGC 346","M8"   ,"OrionL","OrionS"]
LHa    =[4.46e39  ,2.29e39  ,3.98e38   ,7.41e38   ,5.74e39 ,3.98e39 ,4.67e38  ,2.95e37,1e37    ,1e37    ]
SFR    =[5.3e-3   ,2.6e-3   ,3.1e-4    ,1.5e-4    ,1.3e-2  ,1.0e-2  ,0.0021   ,1.0e-4 ,5.3e-5  ,5.3e-5  ]
n      =[3        ,4        ,5         ,8         ,250     ,500     ,100      ,60     ,150     ,4000    ] 
Diam   =[400      ,400      ,160       ,130       ,98.9    ,5       ,64       , 25    ,5       ,0.6     ]
R      =[200      ,200      ,80        ,65        ,49.4    ,7.5     ,32       ,12.5   ,2.5     ,0.3     ]
Dist   =[840      ,840      ,500       ,500       ,50      ,2.35    ,61.7     ,1.25   ,0.4     ,0.4     ]
siglos =[16.21    ,18.33    ,12.3      ,13.4      ,31.7    ,22.46   ,10.2     ,13.6   ,6       ,6       ]
sigW   =[23.1     ,27.1     ,13.4      ,14.7      ,31.7    ,22.46   ,10.2     ,13.6   ,6       ,6       ]

# Confidence intervals with sigma fixed

sig    =[7.39     ,6.64     ,2.8       ,3.59      ,15.8    ,4.22    ,5.6      ,2.7    ,3.23    ,3.1     ]
r0     =[8.75     ,8.99     ,3.62      ,2.7       ,2.69    ,0.53    ,1.4      ,1.78   ,0.84    ,0.062   ]
r0er   =[0.75     ,0.71     ,0.18      ,0.24      ,0.08    ,0.12    ,0.75     ,0.07   ,0.37    ,0.003   ]
m      =[1.95     ,1.75     ,1.7       ,1.62      , 1.22   ,1.19    ,1.13     ,1.36   ,0.9     ,1.44    ]
mer    =[0.14     ,0.10     ,0.09      ,0.11      ,0.02    ,0.20    ,0.03     ,0.05   ,0.04    ,0.11    ]

# + [markdown] id="rs6r9a_-Fqtm"
# Confidence intervals with sigma float
# -

sig2    =[43.46    ,36.56    ,12.77     ,7.88      ,300.67  ,19.99   ,38.01    ,7.29    ,5.28   ,6.68  ]
sig2er  =[5.05     ,3.36     ,0.69      ,0.72      ,27.97   ,3.08    ,3.69     ,0.72    ,12.3   ,0.53  ]
r02     =[6.9      ,8.04     ,3.61      ,2.78      ,3.4     ,0.54    ,1.89     ,1.78   ,0.6    ,0.042 ]
r02er   =[0.83     ,0.85     ,0.21      ,0.35      ,0.44    ,0.18    ,0.35     ,0.17   ,1.88   ,0.004 ]
m2      =[2.07     ,1.83     ,1.72      ,1.62      ,1.18    ,0.98    ,1.05     ,1.36   ,2.17   ,1.65  ]
m2er    =[0.23     ,0.12     ,0.16      ,0.28      ,0.02    ,0.37    ,0.04     ,0.13   ,1.78   ,0.07  ]

# + id="gqiJOHtcDqBL"
data = pd.DataFrame(
    {
       "Region": Region,
       "LHa": LHa,
       "SFR": SFR,
       "n": n,
       "L [pc]": Diam,
       "R [pc]": R,
       "Dist [kpc]": Dist,
       "sig [km/s]": sig,
       "sig2 [km/s]": sig2,
       "sig2er": sig2er,
       "m": m,
       "m2": m2,
       "mer": mer,
       "m2er": m2er,
       "r0 [pc]": r0,
       "r02 [pc]": r02,
       "r0er": r0er,
       "r02er": r02er,
       "siglos [km/s]": siglos,
      
    },
)
# -

data['sig2 [km/s]']=data['sig2 [km/s]']**0.5

data['sig2er']=data['sig2er']**0.5

# + [markdown] id="gcZdCZEeGOjq"
# Checar que los tipos de las columnas sean adecuadas:

# + colab={"base_uri": "https://localhost:8080/"} id="qHpUNclsFR7r" outputId="6c5d205c-8b4f-46f7-b1be-4d683e51a18f"
data.dtypes

# + colab={"base_uri": "https://localhost:8080/", "height": 394} id="K7q-Sf257_9S" outputId="ebb1d04d-40e4-46f1-8239-bfac59542a8a"
data

# + [markdown] id="0jU-WFLzI7BN"
# ## Change to log scale for most parameters
#
# El analisis de la mayoría de las columnas sería mejor en escala logarítmica porque varían por varios ordenes de magnitud.  Entonces, hacemos una nueva versión de la tabla así. Dejamos la $m$ en escala lineal porque no varía mucho.

# + colab={"base_uri": "https://localhost:8080/", "height": 430} id="ikpoilaIeOCK" outputId="a174005b-3d15-4cbe-a86e-2794486756ac"
cols = data.columns
logdata = data.copy()
for col in cols:
    if col not in ["Region", "m", "mer","m2","m2er","r0er","r02er","sig2er"]:
        logdata[col] = np.round(np.log10(logdata[col]), 2)
        logdata.rename(columns={col: f"log {col}"}, inplace=True)
# Some minor changes to column names
logdata.rename(
    columns={
        "log LHa": "log L(H) [erg s^-1]",
        }, 
    inplace=True)
logdata

# + [markdown] id="RsZZVwFdjtCZ"
# Make the label text bigger on the figures

# + id="xglVwYE0VFBr"
sns.set_context("talk")

# + [markdown] id="GBbSKFIgj61F"
# Repeat the pair plot of correlations between columns with the log-scale quantities. Use color to indicate the distance to the regions.

# + colab={"base_uri": "https://localhost:8080/", "height": 895} id="3bNPqA-OeWoV" outputId="9701fc58-faab-4ff4-bb86-54717aed1f1d"
selected_vars = [ "log L [pc]","log L(H) [erg s^-1]", "log Dist [kpc]", "m", "log r0 [pc]", "log sig [km/s]", "log siglos [km/s]"]
plotdata = logdata[selected_vars].rename(
    columns={
        # Switch column names to use latex formatting to improve axis labels
        "log L [pc]": r"$\log_{10}\ L$ [pc]", 
        "log L(H) [erg s^-1]": r"$\log_{10}\ L(\mathrm{H})$ [erg s$^{-1}$]", 
        "m": "$m$", 
        "log r0 [pc]": r"$\log_{10}\ r_0$ [pc]", 
        "log sig [km/s]": r"$\log_{10}\ \sigma$ [km/s]", 
        "log Dist [kpc]": r"$\log_{10}\ D$ [kpc]",
    },
)

sns.pairplot(plotdata, 
             hue=r"$\log_{10}\ D$ [kpc]",
             plot_kws=dict(alpha=0.75, s=200, edgecolor="k"), 
             diag_kind='hist',
             diag_kws= dict(multiple='stack'),
             );

figname = "strucfunc-correlations"
# Save PDF and JPG versions of the figure
#plt.gcf().savefig(f"{figname}.pdf")
#plt.gcf().savefig(f"{figname}.jpg")

# + [markdown] id="gnKxFarT-GMc"
# ## Correlation coefficients
#
# Calculate [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient), $r$, between each pair of variables:

# + colab={"base_uri": "https://localhost:8080/", "height": 258} id="2-J9Q8ynDJVg" outputId="8a59f000-d9fb-4a22-bbcb-29259cc2f200"
logdata[selected_vars].corr()

# + [markdown] id="zfEKMEmaBm3C"
# So the results for correlations between the basic parameters are what we would expect:
#
# - Size is moderately correlated with luminosity ($r = 0.67$).  The correlation is only weak because size will also depend on ambient density and the age of the region. 
# - Size is highly correlated with distance ($r = 0.92$).  This is just a selection effect due to the fact that we tend to choose regions that have an angular size that matches our instruments. 
# - Luminosity–distance correlation is the weakest ($r = 0.71$), and can be explained as being due to the previous two.
#
#

# + [markdown] id="ibTtRHF9Ev-D"
# The correlations of the structure function parameters with the basic parameters are very interesting:
# - Autocorrelation length scale, $r_0$, is *highly* correlated ($r = 0.96$) with region size, $L$.  Looking at the graph, the relation seems to be approximately linear with $\ell_0 \approx 0.1 L$.  How much of this correlation is real and how much is down to selection effects is something we need to consider carefully.
# - Velocity dispersion, $\sigma$, is well correlated ($r = 0.71$) with luminosity, $L(\mathrm{Ha})$.  
# - The structure function slope $m$ is well correlated ($r = 0.73$) with distance.  This is mainly because $m \approx 1.6$ for all the distant regions observed with TAURUS and ISIS, but is around $m = 1$ for all the nearby regions.  I suspect that this is mainly a data quality issue: we observe very little of the inertial range in the distant regions, so the determination of $m$ is probably not so reliable.  But I am not sure why this would tend to bias $m$ towards high values.  
# -

logdata[selected_vars].corr()

logdata


# + [markdown] id="3HVCWiNtLIMB"
# ## Significance tests
#
# We can calculate the statistical significance of the correlations by using the Student t-statistic.  We test the null hypothesis that a given pair of variables is truly uncorrelated ($r = 0$) for the underlying population and that the observed $r$ for our sample of $N = 9$ sources arises solely by chance.  We want to calculate the $p$-value, which is the probability of obtaining $r$ greater than or equal to the observed value, given that there is no true correlation.  A small value of $p$ means that the null hypothesis can be rejected at a certain confidence level.  

# + [markdown] id="7msUNnoEWFCt"
# First we calculate the t-statistic from the Pearson correlation coefficient:
# $$
# t = r \, \left(\frac{N - 2}{1 - r^2}\right)^{1/2}
# $$
# Then, we use the cumulative distribution function (CDF) of the t-distribution with $N - 1$ degrees of freedom to find the p-value. 

# + id="XpmIkg1GYh_V"
def tstatistic(r, n):
    """Compute Student t statistic for null hypothesis of no correlation
    for an observed Pearson correlation of `r` from `n` samples
    """
    return r*np.sqrt((n - 2) / (1 - r**2))


# + id="8jnDPwdlaAUz"
import scipy.stats


# + id="FH9_8fHjbPAF"
def p_from_t(t, n):
    """Compute 1-sided p-value from Student statistic `t` with `n` observations"""
    # sf is survival function: 1 - CDF
    return scipy.stats.t.sf(t, n-1)
    


# + colab={"base_uri": "https://localhost:8080/", "height": 446} id="ru4y0bC_cAk8" outputId="429694eb-fbfe-4356-9fc7-c8823bf3d33d"
N = 9
rvals = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.779, 0.8, 0.9, 0.908, 0.95, 0.981])
tvals = tstatistic(rvals, N)
pvals = p_from_t(tvals, N)
pd.DataFrame(
    {"r": rvals, "t": tvals, "p": pvals}
)

# + colab={"base_uri": "https://localhost:8080/", "height": 332} id="LOdQN93OdF-w" outputId="b8f2ecc1-f0b2-491f-ac34-b537f25b8735"
fig, ax = plt.subplots()
for N in 3, 5, 9:
    rvals = np.linspace(0.0, 1.0, 100, endpoint=False)
    tvals = tstatistic(rvals, N)
    pvals = p_from_t(tvals, N)
    ax.plot(rvals, pvals, label=f"N = {N}")
for p0 in 0.05, 0.01, 0.001:
    ax.axhline(p0, color="k", ls="--", lw=0.5)
    ax.text(1.07, p0, f"{100*(1-p0):.1f}%", 
            va="center", ha="center", fontsize="xx-small", 
            bbox={"color": "w"})
ax.legend(title="# of samples")
ax.set(
    xlabel="Sample Pearson correlation, $r$",
    ylabel="$p$-value",
    yscale="log",
    xlim=[-0.05, 1.15],
    ylim=[1.1e-5, 1.1],
);

# + [markdown] id="Pv_EECxBvyeg"
# I have marked confidence levels for nominal $p$-values of 0.05, 0.01, and 0.001, which are often used to judge significance.   This implies that all the correlations that we listed above are highly significant.  For instance, the correlation of $\sigma$ with luminosity, with $r=0.77$, has $p = 0.011$, so close to the 99% confidence level. 
#
# Despite the good correlation, there is a factor of 4 difference in $\sigma$ between Carina and 30~Dor, despite similar luminosities.  **Need to check these luminostiy values**. 
# -

logdata

logdata['mer']=(data['mer']/data['m'])*0.434

logdata['m2er']=(data['m2er']/data['m2'])*0.434

logdata['r0er']=(data['r0er']/data['r0 [pc]'])*0.434

logdata['r02er']=(data['r02er']/data['r02 [pc]'])*0.434

logdata['sig2er']=(data['sig2er']/data['sig2 [km/s]'])*0.434

logdata



# r0 vs m

X, Xe, Y, Ye = [logdata[_] for _ in ['log r0 [pc]', 'r0er','m', 'mer']]

Y=np.log10(Y)

# +
fig, ax = plt.subplots(figsize=(10, 10))
ax.errorbar(X, Y, xerr=Xe, yerr=Ye, ls=" ", alpha=0.3)
ax.scatter(X, Y, marker=".", s=20/np.hypot(Xe, Ye))

#ax.set(
#    xlim=[-0.2, 0.8], ylim=[-0.2, 0.8],
#    xlabel=r"$\log_{10}\, \Pi$", ylabel=r"$\log_{10}\, \Lambda$",
#)
#ax.set_aspect("equal")
#sns.despine()
# -

lm = linmix.LinMix(X[:-1], Y[:-1], Xe[:-1], Ye[:-1], K=2)

lm.run_mcmc()

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
dfchain

dfchain.describe()

pearsonr(X, Y)

pd.DataFrame({"X": X, "Xe": Xe, "Y": Y, "Ye": Ye}).describe()

# +
vmin, vmax = -3.0, 1.0
xgrid = np.linspace(vmin, vmax, 200)


fig, ax = plt.subplots(figsize=(10, 10))

ax.errorbar(X, Y, xerr=Xe, yerr=Ye, ls=" ", elinewidth=0.4, alpha=1.0, c="k")
ax.scatter(X, Y, marker=".", s=20/np.hypot(Xe, Ye))
# The original fit
ax.plot(xgrid, dfchain["alpha"].mean() + xgrid*dfchain["beta"].mean(), 
        '-', c="k")
for samp in lm.chain[::5]:
    ax.plot(xgrid, samp["alpha"] + xgrid*samp["beta"], 
        '-', c="r", alpha=0.2, lw=0.1)
ax.set(
    xlim=[-1.5, 1.1],
    ylim=[-0.1, 0.4],
    xlabel=r"Velocity correlation length: $\log_{10} r_0$",
    ylabel=r"Structure function slope: $\log_{10} m$",
)
# -

dfchain["alpha"].mean(),dfchain["beta"].mean()

dfchain["alpha"].std(),dfchain["beta"].std()

x,y,z=X,Y,Ye
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())

# Diam(R) vs r0

# Sig Fixed

X, Y, Ye = [logdata[_] for _ in ['log L [pc]', 'log r0 [pc]', 'r0er']]

x,y,z=X,Y,Ye
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())

# Sig Float

X, Y, Ye = [logdata[_] for _ in ['log L [pc]', 'log r02 [pc]', 'r02er']]

x,y,z=X,Y,Ye
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())

# L vs sig

# sig fixed

X, Y = [logdata[_] for _ in ['log L(H) [erg s^-1]', 'log sig [km/s]']]

x,y=X,Y
X = sm.add_constant(x)
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# sig float

X, Y, Ye = [logdata[_] for _ in ['log L(H) [erg s^-1]', 'log sig2 [km/s]', 'sig2er']]

x,y,z=X,Y,Ye
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())

# Distance (D) vs m

# Sigma fixed

X, Y, Ye = [logdata[_] for _ in ['log Dist [kpc]', 'm', 'mer']]

x,y,z=X,Y,Ye
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())

# Sigma Float

X, Y, Ye = [logdata[_] for _ in ['log Dist [kpc]', 'm2', 'm2er']]

x,y,z=X,Y,Ye
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())

# sig vs siglos

X, Y, Ye = [logdata[_] for _ in ['log siglos [km/s]', 'log sig2 [km/s]', 'sig2er']]

x,y,z=X,Y,Ye
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())

# L vs siglos

Moiseev=pd.read_csv('DataOthers//Moiseev2015.csv')
Ostin=pd.read_csv('DataOthers//Ostin2001.csv')
Blasco=pd.read_csv('DataOthers//Blasco2013.csv')
Rozas=pd.read_csv('DataOthers//Rozas2006.csv')
Ars=pd.read_csv('DataOthers//ArsRoy1986.csv')
Wis=pd.read_csv('DataOthers//Wis2012.csv')
Gal=pd.read_csv('DataOthers//Gallagher1983.csv')

# +
fig, ax=plt.subplots(figsize=(9,9))

plt.scatter(Gal.L,Gal.sig,label='Gallagher 1983',marker='x',alpha=0.85,color='dimgray')
plt.scatter(Ars.L,10**Ars.sig,label='Arsenault 1988',marker='+',alpha=0.85,color='dimgray')
plt.scatter(Ostin.L,Ostin.sig,label='Ostin 2001',marker='o',alpha=0.95,color='darkgray')
plt.scatter(Rozas.L,10**(Rozas.sig),label='Rozas 2006',marker='.',alpha=0.95,color='darkgray')
plt.scatter(Wis.L,Wis.sig,label='Wisnioski 2012',marker='s',alpha=0.75,color='silver')
plt.scatter(Blasco.L,Blasco.sig,label='Blasco 2013',marker='D',alpha=0.75,color='silver')
plt.scatter(Moiseev.L,Moiseev.sig,label='Moiseev 2015',marker='^',alpha=0.75,color='silver')

plt.scatter(logdata['log L(H) [erg s^-1]'],10**(logdata['log sig [km/s]']),marker='o',label='SigPOS',color='black',s=(logdata['log Dist [kpc]']+1.0)*70)
plt.scatter(logdata['log L(H) [erg s^-1]'],10**(logdata['log siglos [km/s]']),marker='v',label='SigLOS',color='black',s=(logdata['log Dist [kpc]']+1.0)*70)

plt.yscale('log')

ax.set(
#    ylim  = [36, 43],
#    xlim  = [1, 150],
)
#ax.set_facecolor('whitesmoke')
ax.set(xlabel='Log(L$_{Hα}$) [erg/s]', ylabel='$σ$ [km s$^{-1}$]')
plt.legend()
#fig.savefig('SFplots//lvss.pdf', bbox_inches='tight')


# +
fig, ax=plt.subplots(figsize=(9,9))

plt.scatter(Gal.L,Gal.sig,label='Gallagher 1983',marker='x',alpha=0.85,color='dimgray')
plt.scatter(Ars.L,10**Ars.sig,label='Arsenault 1988',marker='+',alpha=0.85,color='dimgray')
plt.scatter(Ostin.L,Ostin.sig,label='Ostin 2001',marker='o',alpha=0.95,color='darkgray')
plt.scatter(Rozas.L,10**(Rozas.sig),label='Rozas 2006',marker='.',alpha=0.95,color='darkgray')
plt.scatter(Wis.L,Wis.sig,label='Wisnioski 2012',marker='s',alpha=0.75,color='silver')
plt.scatter(Blasco.L,Blasco.sig,label='Blasco 2013',marker='D',alpha=0.75,color='silver')
plt.scatter(Moiseev.L,Moiseev.sig,label='Moiseev 2015',marker='^',alpha=0.75,color='silver')

#plt.scatter(logdata['log L(H) [erg s^-1]'],10**(logdata['log sig [km/s]']),marker='o',label='SigPOS',color='black',s=(logdata['log Dist [kpc]']+1.0)*70)
plt.scatter(logdata['log L(H) [erg s^-1]'],10**(logdata['log siglos [km/s]']),marker='v',label='SigLOS',color='black',s=(logdata['log Dist [kpc]']+1.0)*70)

plt.yscale('log')

ax.set(
#    ylim  = [36, 43],
#    xlim  = [1, 150],
)
#ax.set_facecolor('whitesmoke')
ax.set(xlabel='Log(L$_{Hα}$) [erg/s]', ylabel='$σ$ [km s$^{-1}$]')
plt.legend()
#fig.savefig('SFplots//lvss.pdf', bbox_inches='tight')
# -

globalL= pd.concat([Moiseev.L, Ostin.L.dropna(),Blasco.L.dropna(),Rozas.L.dropna(),Ars.L.dropna(),Wis.L.dropna(),Gal.L.dropna()], axis=0)
globalS= pd.concat([Moiseev.sig, Ostin.sig.dropna(),Blasco.sig.dropna(),10**Rozas.sig.dropna(),10**Ars.sig.dropna(),Wis.sig.dropna(),Gal.sig.dropna()],  axis=0)
GL=np.concatenate((np.array(globalL), np.array(logdata['log L(H) [erg s^-1]'])))
GS=np.concatenate((np.array(globalS), np.array(10**(logdata['log siglos [km/s]']))))

# +
fig, ax=plt.subplots(figsize=(4,4))


plt.scatter(GL, np.log10(GS),alpha=0.45)

x2=np.logspace(1.567,1.63)
y2=-4.4928+0.1472*x2
plt.plot(x2,y2, color='gray')

ax.set(xlabel='Log(L$_{Hα}$) [erg/s]', ylabel='Log$σ$ [km s$^{-1}$]')

# -

GSer = (GS*.1)/GS

GLer = (GL*.1)/GL

X, Xe, Y, Ye = [GL, GLer, np.log10(GS), GSer]

fig, ax = plt.subplots(figsize=(10, 10))
ax.errorbar(X, Y, xerr=Xe, yerr=Ye, ls=" ", alpha=0.3)
ax.scatter(X, Y, marker=".", s=20/np.hypot(Xe, Ye))
#plt.xscale('log')
#plt.yscale('log')

lm = linmix.LinMix(X, Y, Xe, Ye, K=2)

lm.run_mcmc()

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
dfchain

dfchain.describe()

pearsonr(X, Y)

pd.DataFrame({"X": X, "Xe": Xe, "Y": Y, "Ye": Ye}).describe()

# +
vmin, vmax = 35, 45
xgrid = np.linspace(vmin, vmax, 200)


fig, ax = plt.subplots(figsize=(10, 10))

ax.errorbar(X, Y, xerr=Xe, yerr=Ye, ls=" ", elinewidth=0.4, alpha=1.0, c="k")
ax.scatter(X, Y, marker=".", s=20/np.hypot(Xe, Ye))
# The original fit
ax.plot(xgrid, dfchain["alpha"].mean() + xgrid*dfchain["beta"].mean(), 
        '-', c="k")
for samp in lm.chain[::20]:
    ax.plot(xgrid, samp["alpha"] + xgrid*samp["beta"], 
        '-', c="r", alpha=0.2, lw=0.1)
# -

dfchain["alpha"].mean(),dfchain["beta"].mean()

dfchain["alpha"].std(),dfchain["beta"].std()

x,y,z=X,Y,Ye
X = sm.add_constant(x)
model = sm.WLS(y, X,weights=1./(z**2))
results = model.fit()
print(results.summary())

Y, Ye, X, Xe = [GL, GLer, np.log10(GS), GSer]

fig, ax = plt.subplots(figsize=(10, 10))
ax.errorbar(X, Y, xerr=Xe, yerr=Ye, ls=" ", alpha=0.3)
ax.scatter(X, Y, marker=".", s=20/np.hypot(Xe, Ye))

lm = linmix.LinMix(X, Y, Xe, Ye, K=2)

lm.run_mcmc()

dfchain = pd.DataFrame.from_records(
    lm.chain.tolist(), 
    columns=lm.chain.dtype.names
)
dfchain

dfchain.describe()

pearsonr(X, Y)

pd.DataFrame({"X": X, "Xe": Xe, "Y": Y, "Ye": Ye}).describe()

# +
vmin, vmax = 0.5, 2
xgrid = np.linspace(vmin, vmax, 200)


fig, ax = plt.subplots(figsize=(10, 10))

ax.errorbar(X, Y, xerr=Xe, yerr=Ye, ls=" ", elinewidth=0.4, alpha=1.0, c="k")
ax.scatter(X, Y, marker=".", s=20/np.hypot(Xe, Ye))
# The original fit
ax.plot(xgrid, dfchain["alpha"].mean() + xgrid*dfchain["beta"].mean(), 
        '-', c="k")
for samp in lm.chain[::20]:
    ax.plot(xgrid, samp["alpha"] + xgrid*samp["beta"], 
        '-', c="r", alpha=0.4, lw=0.1)

# -

dfchain["alpha"].mean(),dfchain["beta"].mean()

dfchain["alpha"].std(),dfchain["beta"].std()





# + id="ktZZMIJNgkz8"
print("--- %s seconds ---" % (time.time()-start_time))

