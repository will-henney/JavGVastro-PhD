import numpy as np
import scipy.fftpack
from scipy import ndimage
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import os

#SABRE
#Statistical Analysis of Bidimensional REgions

#Updated

#14.04.20- ErrrSlope to SF dataframe

#Pre:
#Add + 's' algorithms for sample analysis (instead of all data analysis).
#ddof=1 in the statiscal analysis instead of 0: sosfs,sofsnormd

#sosf
#sosfs
#sosfnorm
#sosfnorms
#sosfw
#sosfwx
#sosfw2
#acf

##############################################################################
def sosf(data):

    pc=1.0

    df1=data
    df1n=df1.to_numpy()

    xl=[row[0] for row in df1n]
    yl=[row[1] for row in df1n]
    fl=[row[2] for row in df1n]

    x1=[xl]
    y1=[yl]
    f1=[fl]

    x=list(map(list, zip(*x1)))
    y=list(map(list, zip(*y1)))
    f=list(map(list, zip(*f1)))

    sig2=2*(data.RV.var())
    fm=np.nanmean(f)
    fv=np.nanvar(f)
    fs=np.nanstd(f)
    l=len(f)
    S=[[fm],[fv],[fs],[l]]

    a=[[0]*(l) for i in range(l)]
    b=[[0]*(l) for i in range(l)]
    c=[[0]*(2) for i in range(l*l)]

    #Second Order Structure Function Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                a[i][j]=(((f[j][0])-(f[i][0]))**2)

    #Coord Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                b[i][j]=((x[i][0]-x[j][0])**2+(y[i][0]-y[j][0])**2)**0.5

    #LxL to Lx2
    #Main Matrix  [(L*L)x2]
    for j in range (l):
        for i in range (l):
                c[i+j*(l)][1]=a[i][j]

    for j in range (l):
        for i in range (l):
                c[i+j*(l)][0]=b[i][j]

    #Filter: Greater than "0"
    d=np.asarray(c)
    ind=np.squeeze(d[:,1])>0.0
    e=d[ind]
    #Pandas Stuff
    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Main Filter Group
    m=df.lag[1]-df.lag[0]#Lag step
    n=int(df.lag.max()//m)-1#Number of points
    #Grouping points
    dfx=dict()
    for i in range(n):
        p=0+i*(m+0.001)#rs
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    lgp=dict()
    lpc=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        lpc[i]=lgp[i]*pc
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i]-1)
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i]-1)


        g=[[0]*(8) for i in range(n)]

    for i in range(n):
        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=lpc[i]
        g[i][6]=dl[i]
        g[i][7]=derx[i]

    mx=[row[0] for row in g]
    nx=[row[1] for row in g]
    o=[row[2] for row in g]
    mpc=[row[5] for row in g]

    #Exponent

    xa=[mx]
    xe=list(map(list, zip(*xa)))
    ya=[nx]
    ye=list(map(list, zip(*ya)))

    L=len(xe)

    me=[[0]*(1) for i in range(L)]
    ne=[[0]*(1) for i in range(L)]
    oe=[[0]*(1) for i in range(L)]
    pe=[[0]*(1) for i in range(L)]
    qe=[[0]*(1) for i in range(L)]
    re=[[0]*(1) for i in range(L)]
    se=[[0]*(1) for i in range(L)]
    te=[[0]*(1) for i in range(L)]
    ue=[[0]*(1) for i in range(L)]
    we=[[0]*(1) for i in range(L)]

    for i in range (L):
        me[i][0]=math.log10(xe[i][0])
        ne[i][0]=math.log10(ye[i][0])
        oe[i][0]=(me[i][0]*ne[i][0])
        pe[i][0]=(me[i][0])**2

    def sumColumn(a):
        return [sum(col) for col in zip(*a)]

    for i in range (L+1):
        if i>0:
            qe[i-1]=sumColumn(me[:i])
            re[i-1]=sumColumn(ne[:i])
            se[i-1]=sumColumn(oe[:i])
            te[i-1]=sumColumn(pe[:i])
            ue[i-1][0]=len(x[:i])
        else:
            we[0][0]=0

    for i in range (L):
        if i>0:
            we[i][0]=((qe[i][0]*re[i][0])-(ue[i][0]*se[i][0]))/((qe[i][0])**2-ue[i][0]*te[i][0])
        else:
            we[0][0]=0

    A=[xe,ye,me,ne,oe,pe,qe,re,se,te,ue,we]
    dfe=pd.DataFrame(A)
    dfet=dfe.transpose()
    dfet.columns=['Lag.x','Nmqvd.y','Logx','Logy','Logx*Logy','Logx^2','SumLogx','SumLogy','Sum','Sum','N','Exp']

    exp=we
    ft=np.append(g,we, axis=1)
    sf=pd.DataFrame(ft)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','Lag pc','# P','ErrX','Exp']

    #Error Slope
    L=len(sf)
    m1=[[0]*(1) for i in range(L)]
    m2=[[0]*(1) for i in range(L)]
    ap=[[0]*(1) for i in range(L)]
    ee=[[0]*(1) for i in range(L)]

    for n in range(0,L):

        a=sf.Nmqvd[n]+sf.ErrY[n]
        b=sf.Nmqvd[0]-sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m1[n]=g

        a=sf.Nmqvd[n]-sf.ErrY[n]
        b=sf.Nmqvd[0]+sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m2[n]=g

        ap0=[np.array(m1[n]),np.array(m2[n]),sf.Exp[n]]
        ap[n]=np.mean(ap0)

        ee0=[m1[n],m2[n],sf.Exp[n]]
        ee[n]=np.std(ee0)/np.sqrt(len(ee0))

    sf['Exp1']=m1
    sf['Exp2']=m2
    sf['Alpha']=ap
    sf['ErrT']=ee
    sf.round(3)

    fig, ax=plt.subplots()
    ax.errorbar(sf['Lag'],sf['Nmqvd'],xerr=sf.ErrX,yerr=sf.ErrY, fmt="o",color='b', ecolor='k', alpha=0.45, markersize=7.5)
    ax.set(xscale='log',yscale='log')
    ax.axhline(y=sig2, color='k', linestyle='--')
    ax.set(xlabel='lag [arcsec]', ylabel=r'S(l) [km$^{2}$/s$^{2}$]')
    plt.tick_params(which='both', labelright=True, direction='in', right=True,  top=True)
    plt.grid(which='minor')

    return sf
##############################################################################
def sosfs(data):

    pc=1.0

    df1=data
    df1n=df1.to_numpy()

    xl=[row[0] for row in df1n]
    yl=[row[1] for row in df1n]
    fl=[row[2] for row in df1n]

    x1=[xl]
    y1=[yl]
    f1=[fl]

    x=list(map(list, zip(*x1)))
    y=list(map(list, zip(*y1)))
    f=list(map(list, zip(*f1)))

    sig2=2*(data.RV.var())
    fm=np.nanmean(f)
    fv=np.nanvar(f,ddof=1)
    fs=np.nanstd(f,ddof=1)
    l=len(f)
    S=[[fm],[fv],[fs],[l]]

    a=[[0]*(l) for i in range(l)]
    b=[[0]*(l) for i in range(l)]
    c=[[0]*(2) for i in range(l*l)]

    #Second Order Structure Function Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                a[i][j]=(((f[j][0])-(f[i][0]))**2)

    #Coord Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                b[i][j]=((x[i][0]-x[j][0])**2+(y[i][0]-y[j][0])**2)**0.5

    #LxL to Lx2
    #Main Matrix  [(L*L)x2]
    for j in range (l):
        for i in range (l):
                c[i+j*(l)][1]=a[i][j]

    for j in range (l):
        for i in range (l):
                c[i+j*(l)][0]=b[i][j]

    #Filter: Greater than "0"
    d=np.asarray(c)
    ind=np.squeeze(d[:,1])>0.0
    e=d[ind]
    #Pandas Stuff
    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Main Filter Group
    m=df.lag[1]-df.lag[0]#Lag step
    n=int(df.lag.max()//m)-1#Number of points
    #Grouping points
    dfx=dict()
    for i in range(n):
        p=0+i*(m+0.001)#rs
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    lgp=dict()
    lpc=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        lpc[i]=lgp[i]*pc
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i]-1)
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i]-1)


        g=[[0]*(8) for i in range(n)]

    for i in range(n):
        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=lpc[i]
        g[i][6]=dl[i]
        g[i][7]=derx[i]

    mx=[row[0] for row in g]
    nx=[row[1] for row in g]
    o=[row[2] for row in g]
    mpc=[row[5] for row in g]

    #Exponent

    xa=[mx]
    xe=list(map(list, zip(*xa)))
    ya=[nx]
    ye=list(map(list, zip(*ya)))

    L=len(xe)

    me=[[0]*(1) for i in range(L)]
    ne=[[0]*(1) for i in range(L)]
    oe=[[0]*(1) for i in range(L)]
    pe=[[0]*(1) for i in range(L)]
    qe=[[0]*(1) for i in range(L)]
    re=[[0]*(1) for i in range(L)]
    se=[[0]*(1) for i in range(L)]
    te=[[0]*(1) for i in range(L)]
    ue=[[0]*(1) for i in range(L)]
    we=[[0]*(1) for i in range(L)]

    for i in range (L):
        me[i][0]=math.log10(xe[i][0])
        ne[i][0]=math.log10(ye[i][0])
        oe[i][0]=(me[i][0]*ne[i][0])
        pe[i][0]=(me[i][0])**2

    def sumColumn(a):
        return [sum(col) for col in zip(*a)]

    for i in range (L+1):
        if i>0:
            qe[i-1]=sumColumn(me[:i])
            re[i-1]=sumColumn(ne[:i])
            se[i-1]=sumColumn(oe[:i])
            te[i-1]=sumColumn(pe[:i])
            ue[i-1][0]=len(x[:i])
        else:
            we[0][0]=0

    for i in range (L):
        if i>0:
            we[i][0]=((qe[i][0]*re[i][0])-(ue[i][0]*se[i][0]))/((qe[i][0])**2-ue[i][0]*te[i][0])
        else:
            we[0][0]=0

    A=[xe,ye,me,ne,oe,pe,qe,re,se,te,ue,we]
    dfe=pd.DataFrame(A)
    dfet=dfe.transpose()
    dfet.columns=['Lag.x','Nmqvd.y','Logx','Logy','Logx*Logy','Logx^2','SumLogx','SumLogy','Sum','Sum','N','Exp']

    exp=we
    ft=np.append(g,we, axis=1)
    sf=pd.DataFrame(ft)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','Lag pc','# P','ErrX','Exp']

    #Error Slope
    L=len(sf)
    m1=[[0]*(1) for i in range(L)]
    m2=[[0]*(1) for i in range(L)]
    ap=[[0]*(1) for i in range(L)]
    ee=[[0]*(1) for i in range(L)]

    for n in range(0,L):

        a=sf.Nmqvd[n]+sf.ErrY[n]
        b=sf.Nmqvd[0]-sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m1[n]=g

        a=sf.Nmqvd[n]-sf.ErrY[n]
        b=sf.Nmqvd[0]+sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m2[n]=g

        ap0=[np.array(m1[n]),np.array(m2[n]),sf.Exp[n]]
        ap[n]=np.mean(ap0)

        ee0=[m1[n],m2[n],sf.Exp[n]]
        ee[n]=np.std(ee0)/np.sqrt(len(ee0))

    sf['Exp1']=m1
    sf['Exp2']=m2
    sf['Alpha']=ap
    sf['ErrT']=ee
    sf.round(3)

    fig, ax=plt.subplots()
    ax.errorbar(sf['Lag'],sf['Nmqvd'],xerr=sf.ErrX,yerr=sf.ErrY, fmt="o",color='b', ecolor='k', alpha=0.45, markersize=7.5)
    ax.set(xscale='log',yscale='log')
    ax.axhline(y=sig2, color='k', linestyle='--')
    ax.set(xlabel='lag [arcsec]', ylabel=r'S(l) [km$^{2}$/s$^{2}$]')
    plt.tick_params(which='both', labelright=True, direction='in', right=True,  top=True)
    plt.grid(which='minor')

    return sf
################################################################################
def sosfnorm(data):

    pc=1.0

    df1=data
    #df1=abs(data)
    df1n=df1.to_numpy()

    xl=[row[0] for row in df1n]
    yl=[row[1] for row in df1n]
    fl=[row[2] for row in df1n]

    x1=[xl]
    y1=[yl]
    f1=[fl]

    x=list(map(list, zip(*x1)))
    y=list(map(list, zip(*y1)))
    f=list(map(list, zip(*f1)))

    sig2=2#Change to header third column
    fm=np.nanmean(f)
    fv=np.nanvar(f)
    fs=np.nanstd(f)
    l=len(f)
    S=[[fm],[fv],[fs],[l]]

    a=[[0]*(l) for i in range(l)]
    b=[[0]*(l) for i in range(l)]
    c=[[0]*(2) for i in range(l*l)]

    #normalized Second Order Structure Function Matrix [LxL]
    for i in range(l):
        for j in range(l):
          if i > j:
              a[i][j]=(((f[j][0]-fm)-(f[i][0]-fm))**2)/(fv)

    #Coord Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                b[i][j]=((x[i][0]-x[j][0])**2+(y[i][0]-y[j][0])**2)**0.5

    #LxL to Lx2
    #Main Matrix  [(L*L)x2]
    for j in range (l):
        for i in range (l):
                c[i+j*(l)][1]=a[i][j]

    for j in range (l):
        for i in range (l):
                c[i+j*(l)][0]=b[i][j]

    #Filter: Greater than "0"
    d=np.asarray(c)
    ind=np.squeeze(d[:,1])>0.0
    e=d[ind]
    #Pandas Stuff
    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Main Filter Group
    m=df.lag[1]-df.lag[0]#Lag step
    n=int(df.lag.max()//m)-1#Number of points
    #Grouping points
    dfx=dict()
    for i in range(n):
        p=0+i*(m+0.001)#rs
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    lgp=dict()
    lpc=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        lpc[i]=lgp[i]*pc
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()#
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i]-1)
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i]-1)


        g=[[0]*(8) for i in range(n)]

    for i in range(n):
        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=lpc[i]
        g[i][6]=dl[i]
        g[i][7]=derx[i]

    mx=[row[0] for row in g]
    nx=[row[1] for row in g]
    o=[row[2] for row in g]
    mpc=[row[5] for row in g]

    #Exponent

    xa=[mx]
    xe=list(map(list, zip(*xa)))
    ya=[nx]
    ye=list(map(list, zip(*ya)))

    L=len(xe)

    me=[[0]*(1) for i in range(L)]
    ne=[[0]*(1) for i in range(L)]
    oe=[[0]*(1) for i in range(L)]
    pe=[[0]*(1) for i in range(L)]
    qe=[[0]*(1) for i in range(L)]
    re=[[0]*(1) for i in range(L)]
    se=[[0]*(1) for i in range(L)]
    te=[[0]*(1) for i in range(L)]
    ue=[[0]*(1) for i in range(L)]
    we=[[0]*(1) for i in range(L)]

    for i in range (L):
        me[i][0]=math.log10(xe[i][0])
        ne[i][0]=math.log10(ye[i][0])
        oe[i][0]=(me[i][0]*ne[i][0])
        pe[i][0]=(me[i][0])**2

    def sumColumn(a):
        return [sum(col) for col in zip(*a)]

    for i in range (L+1):
        if i>0:
            qe[i-1]=sumColumn(me[:i])
            re[i-1]=sumColumn(ne[:i])
            se[i-1]=sumColumn(oe[:i])
            te[i-1]=sumColumn(pe[:i])
            ue[i-1][0]=len(x[:i])
        else:
            we[0][0]=0

    for i in range (L):
        if i>0:
            we[i][0]=((qe[i][0]*re[i][0])-(ue[i][0]*se[i][0]))/((qe[i][0])**2-ue[i][0]*te[i][0])
        else:
            we[0][0]=0

    A=[xe,ye,me,ne,oe,pe,qe,re,se,te,ue,we]
    dfe=pd.DataFrame(A)
    dfet=dfe.transpose()
    dfet.columns=['Lag.x','Nmqvd.y','Logx','Logy','Logx*Logy','Logx^2','SumLogx','SumLogy','Sum','Sum','N','Exp']

    exp=we
    ft=np.append(g,we, axis=1)
    sf=pd.DataFrame(ft)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','Lag pc','# P','ErrX','Exp']

    #Error Slope
    L=len(sf)
    m1=[[0]*(1) for i in range(L)]
    m2=[[0]*(1) for i in range(L)]
    ap=[[0]*(1) for i in range(L)]
    ee=[[0]*(1) for i in range(L)]

    for n in range(0,L):

        a=sf.Nmqvd[n]+sf.ErrY[n]
        b=sf.Nmqvd[0]-sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m1[n]=g

        a=sf.Nmqvd[n]-sf.ErrY[n]
        b=sf.Nmqvd[0]+sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m2[n]=g

        ap0=[np.array(m1[n]),np.array(m2[n]),sf.Exp[n]]
        ap[n]=np.mean(ap0)

        ee0=[m1[n],m2[n],sf.Exp[n]]
        ee[n]=np.std(ee0)/np.sqrt(len(ee0))

    sf['Exp1']=m1
    sf['Exp2']=m2
    sf['Alpha']=ap
    sf['ErrT']=ee
    sf.round(3)

    fig, ax=plt.subplots()
    ax.errorbar(sf['Lag'],sf['Nmqvd'],xerr=sf.ErrX,yerr=sf.ErrY, fmt="o",color='b', ecolor='k', alpha=0.45, markersize=7.5)
    ax.set(xscale='log',yscale='log')
    ax.axhline(y=sig2, color='k', linestyle='--')
    ax.set(xlabel='lag [arcsec]', ylabel=r'S$_{2}$(l) [-]')
    plt.tick_params(which='both', labelright=True, direction='in', right=True,  top=True)
    plt.grid(which='minor')

    return sf
###############################################################################
def sosfnorms(data):

    pc=1.0

    df1=data
    #df1=abs(data)
    df1n=df1.to_numpy()

    xl=[row[0] for row in df1n]
    yl=[row[1] for row in df1n]
    fl=[row[2] for row in df1n]

    x1=[xl]
    y1=[yl]
    f1=[fl]

    x=list(map(list, zip(*x1)))
    y=list(map(list, zip(*y1)))
    f=list(map(list, zip(*f1)))

    sig2=2#Change to header third column
    fm=np.nanmean(f)
    fv=np.nanvar(f,ddof=1)
    fs=np.nanstd(f,ddof=1)
    l=len(f)
    S=[[fm],[fv],[fs],[l]]

    a=[[0]*(l) for i in range(l)]
    b=[[0]*(l) for i in range(l)]
    c=[[0]*(2) for i in range(l*l)]

    #normalized Second Order Structure Function Matrix [LxL]
    for i in range(l):
        for j in range(l):
          if i > j:
              a[i][j]=(((f[j][0]-fm)-(f[i][0]-fm))**2)/(fv)

    #Coord Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                b[i][j]=((x[i][0]-x[j][0])**2+(y[i][0]-y[j][0])**2)**0.5

    #LxL to Lx2
    #Main Matrix  [(L*L)x2]
    for j in range (l):
        for i in range (l):
                c[i+j*(l)][1]=a[i][j]

    for j in range (l):
        for i in range (l):
                c[i+j*(l)][0]=b[i][j]

    #Filter: Greater than "0"
    d=np.asarray(c)
    ind=np.squeeze(d[:,1])>0.0
    e=d[ind]
    #Pandas Stuff
    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Main Filter Group
    m=df.lag[1]-df.lag[0]#Lag step
    n=int(df.lag.max()//m)-1#Number of points
    #Grouping points
    dfx=dict()
    for i in range(n):
        p=0+i*(m+0.001)#rs
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    lgp=dict()
    lpc=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        lpc[i]=lgp[i]*pc
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()#
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i]-1)
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i]-1)


        g=[[0]*(8) for i in range(n)]

    for i in range(n):
        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=lpc[i]
        g[i][6]=dl[i]
        g[i][7]=derx[i]

    mx=[row[0] for row in g]
    nx=[row[1] for row in g]
    o=[row[2] for row in g]
    mpc=[row[5] for row in g]

    #Exponent

    xa=[mx]
    xe=list(map(list, zip(*xa)))
    ya=[nx]
    ye=list(map(list, zip(*ya)))

    L=len(xe)

    me=[[0]*(1) for i in range(L)]
    ne=[[0]*(1) for i in range(L)]
    oe=[[0]*(1) for i in range(L)]
    pe=[[0]*(1) for i in range(L)]
    qe=[[0]*(1) for i in range(L)]
    re=[[0]*(1) for i in range(L)]
    se=[[0]*(1) for i in range(L)]
    te=[[0]*(1) for i in range(L)]
    ue=[[0]*(1) for i in range(L)]
    we=[[0]*(1) for i in range(L)]

    for i in range (L):
        me[i][0]=math.log10(xe[i][0])
        ne[i][0]=math.log10(ye[i][0])
        oe[i][0]=(me[i][0]*ne[i][0])
        pe[i][0]=(me[i][0])**2

    def sumColumn(a):
        return [sum(col) for col in zip(*a)]

    for i in range (L+1):
        if i>0:
            qe[i-1]=sumColumn(me[:i])
            re[i-1]=sumColumn(ne[:i])
            se[i-1]=sumColumn(oe[:i])
            te[i-1]=sumColumn(pe[:i])
            ue[i-1][0]=len(x[:i])
        else:
            we[0][0]=0

    for i in range (L):
        if i>0:
            we[i][0]=((qe[i][0]*re[i][0])-(ue[i][0]*se[i][0]))/((qe[i][0])**2-ue[i][0]*te[i][0])
        else:
            we[0][0]=0

    A=[xe,ye,me,ne,oe,pe,qe,re,se,te,ue,we]
    dfe=pd.DataFrame(A)
    dfet=dfe.transpose()
    dfet.columns=['Lag.x','Nmqvd.y','Logx','Logy','Logx*Logy','Logx^2','SumLogx','SumLogy','Sum','Sum','N','Exp']

    exp=we
    ft=np.append(g,we, axis=1)
    sf=pd.DataFrame(ft)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','Lag pc','# P','ErrX','Exp']

    #Error Slope
    L=len(sf)
    m1=[[0]*(1) for i in range(L)]
    m2=[[0]*(1) for i in range(L)]
    ap=[[0]*(1) for i in range(L)]
    ee=[[0]*(1) for i in range(L)]

    for n in range(0,L):

        a=sf.Nmqvd[n]+sf.ErrY[n]
        b=sf.Nmqvd[0]-sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m1[n]=g

        a=sf.Nmqvd[n]-sf.ErrY[n]
        b=sf.Nmqvd[0]+sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m2[n]=g

        ap0=[np.array(m1[n]),np.array(m2[n]),sf.Exp[n]]
        ap[n]=np.mean(ap0)

        ee0=[m1[n],m2[n],sf.Exp[n]]
        ee[n]=np.std(ee0)/np.sqrt(len(ee0))

    sf['Exp1']=m1
    sf['Exp2']=m2
    sf['Alpha']=ap
    sf['ErrT']=ee
    sf.round(3)

    fig, ax=plt.subplots()
    ax.errorbar(sf['Lag'],sf['Nmqvd'],xerr=sf.ErrX,yerr=sf.ErrY, fmt="o",color='b', ecolor='k', alpha=0.45, markersize=7.5)
    ax.set(xscale='log',yscale='log')
    ax.axhline(y=sig2, color='k', linestyle='--')
    ax.set(xlabel='lag [arcsec]', ylabel=r'S$_{2}$(l) [-]')
    plt.tick_params(which='both', labelright=True, direction='in', right=True,  top=True)
    plt.grid(which='minor')

    return sf

##############################################################################
def sosfw(data):
    pc=1.0

    df1=data
    df1n=df1.to_numpy()
    xl=[row[0] for row in df1n]
    yl=[row[1] for row in df1n]
    fl=[row[2] for row in df1n]
    gl=[row[3] for row in df1n]

    x1=[xl]
    y1=[yl]
    f1=[fl]
    g1=[gl]

    x=list(map(list, zip(*x1)))
    y=list(map(list, zip(*y1)))
    f=list(map(list, zip(*f1)))
    g=list(map(list, zip(*g1)))


    # Weighted Structure Function code

    #StatisticalDataOfTheSampleRVHalpha
    sig2=2*(data.RV.var())#Sigmax2
    fm=np.nanmean(f)#RVMean
    fv=np.nanvar(f,ddof=1)#RVvariance
    fs=np.nanstd(f,ddof=1)#RVstandarddev
    l=len(f)#Lengt
    S=[[fm],[fv],[fs],[l]]#TableWithAllOfTheAbove


    a=[[0]*(l) for i in range(l)]#MatrixForStatisticalFunctionComputation
    b=[[0]*(l) for i in range(l)]#Matrix of Separations-Lags
    c=[[0]*(2) for i in range(l*l)]#Main Matrix [a,b]

    a1=[[0]*(l) for i in range(l)]#MatrixForStatisticalFunctionComputation
    c1=[[0]*(1) for i in range(l*l)]#Main Matrix [a,b]

    for i in range(l):
        for j in range(l):
            if i > j:
                a1[i][j]=(g[j][0]*g[i][0])

    for j in range (l):
        for i in range (l):
            c1[i+j*(l)][0]=a1[i][j]

    W=np.sum(g)

    # Weighted Second Order Structure Function Matrix [LxL] I
    for i in range(l):
        for j in range(l):
            if i > j:
                a[i][j]=(((f[j][0]-f[i][0])**2)*(g[j][0]*g[i][0]))/(fv*W)


    #Coord Matrix [LxL]
    for i in range (l):
        for j in range (l):
            if i > j:
                b[i][j]=((x[i][0]-x[j][0])**2+(y[i][0]-y[j][0])**2)**0.5

    #LxL to Lx2
    #Main Matrix  [(L*L)x2]
    for j in range (l):
        for i in range (l):
            c[i+j*(l)][1]=a[i][j]

    for j in range (l):
        for i in range (l):
            c[i+j*(l)][0]=b[i][j]

    #Filter: Greater than "0"
    d=np.asarray(c)
    ind=np.squeeze(d[:,1])>0.0
    e=d[ind]
    #Pandas Stuff
    df2=pd.DataFrame(e)
    df2.columns=['lag','qvd']#squared velocities difference
    df2.describe()

    # Structure Function Data Groups
    #Main Filter Group
    m=df2.lag[1]-df2.lag[0]#Lag step
    n=int(df2.lag.max()//df2.lag.min())-1#Number of points
    #Grouping points
    dfx=dict()
    for i in range(n):
        p=0+i*(m+0.001)#rs
        q=m+i*m
        dfx[i]=df2[df2['lag'].between(p,q)]

    # Main Matrix Statistical Properties Groups
    lgp=dict()
    lpc=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        lpc[i]=lgp[i]*pc
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i]-1)
            #dery[i]=dfs[i]/tet
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i]-1)

    g=[[0]*(8) for i in range(n)]

    for i in range(n):

        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=lpc[i]
        g[i][6]=dl[i]
        g[i][7]=derx[i]

    mx=[row[0] for row in g]
    nx=[row[1] for row in g]
    o=[row[2] for row in g]
    mpc=[row[5] for row in g]

    # Power law calculation.
    # Index
    xa=[mx]
    xe=list(map(list, zip(*xa)))
    ya=[nx]
    ye=list(map(list, zip(*ya)))

    L=len(xe)

    me=[[0]*(1) for i in range(L)]
    ne=[[0]*(1) for i in range(L)]
    oe=[[0]*(1) for i in range(L)]
    pe=[[0]*(1) for i in range(L)]
    qe=[[0]*(1) for i in range(L)]
    re=[[0]*(1) for i in range(L)]
    se=[[0]*(1) for i in range(L)]
    te=[[0]*(1) for i in range(L)]
    ue=[[0]*(1) for i in range(L)]
    we=[[0]*(1) for i in range(L)]

    for i in range (L):
        me[i][0]=math.log10(xe[i][0])
        ne[i][0]=math.log10(ye[i][0])
        oe[i][0]=(me[i][0]*ne[i][0])
        pe[i][0]=(me[i][0])**2

    def sumColumn(a):
        return [sum(col) for col in zip(*a)]

    for i in range (L+1):
        if i>0:
            qe[i-1]=sumColumn(me[:i])
            re[i-1]=sumColumn(ne[:i])
            se[i-1]=sumColumn(oe[:i])
            te[i-1]=sumColumn(pe[:i])
            ue[i-1][0]=len(x[:i])
        else:
            we[0][0]=0

    for i in range (L):
        if i>0:
            we[i][0]=((qe[i][0]*re[i][0])-(ue[i][0]*se[i][0]))/((qe[i][0])**2-ue[i][0]*te[i][0])
        else:
            we[0][0]=0


    A=[xe,ye,me,ne,oe,pe,qe,re,se,te,ue,we]
    dfe=pd.DataFrame(A)
    dfet=dfe.transpose()
    dfet.columns=['Lag.x','Nmqvd.y','Logx','Logy','Logx*Logy','Logx^2','SumLogx','SumLogy','Sum','Sum','N','Exp']
    dfet.head()

    exp=we
    ft=np.append(g,we, axis=1)
    sf=pd.DataFrame(ft)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','Lag pc','# P','ErrX','Exp']

    #Error Slope
    L=len(sf)
    m1=[[0]*(1) for i in range(L)]
    m2=[[0]*(1) for i in range(L)]
    ap=[[0]*(1) for i in range(L)]
    ee=[[0]*(1) for i in range(L)]

    for n in range(0,L):

        a=sf.Nmqvd[n]+sf.ErrY[n]
        b=sf.Nmqvd[0]-sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m1[n]=g

        a=sf.Nmqvd[n]-sf.ErrY[n]
        b=sf.Nmqvd[0]+sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m2[n]=g

        ap0=[np.array(m1[n]),np.array(m2[n]),sf.Exp[n]]
        ap[n]=np.mean(ap0)

        ee0=[m1[n],m2[n],sf.Exp[n]]
        ee[n]=np.std(ee0)/np.sqrt(len(ee0))

    sf['Exp1']=m1
    sf['Exp2']=m2
    sf['Alpha']=ap
    sf['ErrT']=ee
    sf.round(3)

    fig, ax=plt.subplots()
    ax.errorbar(sf['Lag'],sf['Nmqvd'],xerr=sf.ErrX,yerr=sf.ErrY, fmt="o",color='b', ecolor='k', alpha=0.45, markersize=7.5)
    ax.set(xscale='log',yscale='log')
    #ax.axhline(y=sig2, color='k', linestyle='--')
    #sgrid = np.logspace(.08, 0.7)
    #ax.plot(sgrid, 1.35*sgrid**(1.25), color="k", lw=0.8)
    ax.set(xlabel='lag [arcsec]', ylabel=r'S$_{2}$W(l)')
    plt.tick_params(which='both', labelright=True, direction='in', right=True,  top=True)
    plt.grid(which='minor')

    return sf
#############################################################################
def sosfwx(data):
    pc=1.0

    df1=data
    df1n=df1.to_numpy()
    xl=[row[0] for row in df1n]
    yl=[row[1] for row in df1n]
    fl=[row[2] for row in df1n]
    gl=[row[3] for row in df1n]

    x1=[xl]
    y1=[yl]
    f1=[fl]
    g1=[gl]

    x=list(map(list, zip(*x1)))
    y=list(map(list, zip(*y1)))
    f=list(map(list, zip(*f1)))
    g=list(map(list, zip(*g1)))


    # Weighted Structure Function code

    #StatisticalDataOfTheSampleRVHalpha
    sig2=2*(data.RV.var())#Sigmax2
    fm=np.nanmean(f)#RVMean
    fv=np.nanvar(f,ddof=1)#RVvariance
    fs=np.nanstd(f,ddof=1)#RVstandarddev
    l=len(f)#Lengt
    S=[[fm],[fv],[fs],[l]]#TableWithAllOfTheAbove


    a=[[0]*(l) for i in range(l)]#MatrixForStatisticalFunctionComputation
    b=[[0]*(l) for i in range(l)]#Matrix of Separations-Lags
    c=[[0]*(2) for i in range(l*l)]#Main Matrix [a,b]

    a1=[[0]*(l) for i in range(l)]#MatrixForStatisticalFunctionComputation
    c1=[[0]*(1) for i in range(l*l)]#Main Matrix [a,b]

    for i in range(l):
        for j in range(l):
            if i > j:
                a1[i][j]=(g[j][0]*g[i][0])

    for j in range (l):
        for i in range (l):
            c1[i+j*(l)][0]=a1[i][j]

    W=np.sum(g)

    # Weighted Second Order Structure Function Matrix [LxL] I
    for i in range(l):
        for j in range(l):
            if i > j:
                a[i][j]=(((f[j][0]-f[i][0])**2)*(g[j][0]*g[i][0]))/(fv*W)


    #Coord Matrix [LxL]
    for i in range (l):
        for j in range (l):
            if i > j:
                b[i][j]=((x[i][0]-x[j][0])**2+(y[i][0]-y[j][0])**2)**0.5

    #LxL to Lx2
    #Main Matrix  [(L*L)x2]
    for j in range (l):
        for i in range (l):
            c[i+j*(l)][1]=a[i][j]

    for j in range (l):
        for i in range (l):
            c[i+j*(l)][0]=b[i][j]

    #Filter: Greater than "0"
    d=np.asarray(c)
    ind=np.squeeze(d[:,1])>0.0
    e=d[ind]
    #Pandas Stuff
    df2=pd.DataFrame(e)
    df2.columns=['lag','qvd']#squared velocities difference
    df2.describe()

    # Structure Function Data Groups
    #Main Filter Group
    m=df2.lag[1]-df2.lag[0]#Lag step
    n=int(df2.lag.max()//df2.lag.min())-1#Number of points
    #Grouping points
    dfx=dict()
    for i in range(n):
        p=0+i*(m+0.001)#rs
        q=m+i*m
        dfx[i]=df2[df2['lag'].between(p,q)]

    # Main Matrix Statistical Properties Groups
    lgp=dict()
    lpc=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        lpc[i]=lgp[i]*pc
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        #if dl[i]>0:
        dery[i]=dfs[i]/np.sqrt(dl[i]-1)
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        #if dl[i]>0:
        derx[i]=dls[i]/np.sqrt(dl[i]-1)

    g=[[0]*(8) for i in range(n)]

    for i in range(n):

        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=lpc[i]
        g[i][6]=dl[i]
        g[i][7]=derx[i]

    mx=[row[0] for row in g]
    nx=[row[1] for row in g]
    o=[row[2] for row in g]
    mpc=[row[5] for row in g]

    # Power law calculation.
    # Index
    xa=[mx]
    xe=list(map(list, zip(*xa)))
    ya=[nx]
    ye=list(map(list, zip(*ya)))

    L=len(xe)

    me=[[0]*(1) for i in range(L)]
    ne=[[0]*(1) for i in range(L)]
    oe=[[0]*(1) for i in range(L)]
    pe=[[0]*(1) for i in range(L)]
    qe=[[0]*(1) for i in range(L)]
    re=[[0]*(1) for i in range(L)]
    se=[[0]*(1) for i in range(L)]
    te=[[0]*(1) for i in range(L)]
    ue=[[0]*(1) for i in range(L)]
    we=[[0]*(1) for i in range(L)]

    for i in range (L):
        me[i][0]=math.log10(xe[i][0])
        ne[i][0]=math.log10(ye[i][0])
        oe[i][0]=(me[i][0]*ne[i][0])
        pe[i][0]=(me[i][0])**2

    def sumColumn(a):
        return [sum(col) for col in zip(*a)]

    for i in range (L+1):
        if i>0:
            qe[i-1]=sumColumn(me[:i])
            re[i-1]=sumColumn(ne[:i])
            se[i-1]=sumColumn(oe[:i])
            te[i-1]=sumColumn(pe[:i])
            ue[i-1][0]=len(x[:i])
        else:
            we[0][0]=0

    for i in range (L):
        if i>0:
            we[i][0]=((qe[i][0]*re[i][0])-(ue[i][0]*se[i][0]))/((qe[i][0])**2-ue[i][0]*te[i][0])
        else:
            we[0][0]=0


    A=[xe,ye,me,ne,oe,pe,qe,re,se,te,ue,we]
    dfe=pd.DataFrame(A)
    dfet=dfe.transpose()
    dfet.columns=['Lag.x','Nmqvd.y','Logx','Logy','Logx*Logy','Logx^2','SumLogx','SumLogy','Sum','Sum','N','Exp']
    dfet.head()

    exp=we
    ft=np.append(g,we, axis=1)
    sf=pd.DataFrame(ft)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','Lag pc','# P','ErrX','Exp']

    #Error Slope
    L=len(sf)
    m1=[[0]*(1) for i in range(L)]
    m2=[[0]*(1) for i in range(L)]
    ap=[[0]*(1) for i in range(L)]
    ee=[[0]*(1) for i in range(L)]

    for n in range(0,L):

        a=sf.Nmqvd[n]+sf.ErrY[n]
        b=sf.Nmqvd[0]-sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m1[n]=g

        a=sf.Nmqvd[n]-sf.ErrY[n]
        b=sf.Nmqvd[0]+sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m2[n]=g

        ap0=[np.array(m1[n]),np.array(m2[n]),sf.Exp[n]]
        ap[n]=np.mean(ap0)

        ee0=[m1[n],m2[n],sf.Exp[n]]
        ee[n]=np.std(ee0)/np.sqrt(len(ee0))

    sf['Exp1']=m1
    sf['Exp2']=m2
    sf['Alpha']=ap
    sf['ErrT']=ee
    sf.round(3)

    fig, ax=plt.subplots()
    ax.errorbar(sf['Lag'],sf['Nmqvd'],xerr=sf.ErrX,yerr=sf.ErrY, fmt="o",color='b', ecolor='k', alpha=0.45, markersize=7.5)
    ax.set(xscale='log',yscale='log')
    #ax.axhline(y=sig2, color='k', linestyle='--')
    #sgrid = np.logspace(.08, 0.7)
    #ax.plot(sgrid, 1.35*sgrid**(1.25), color="k", lw=0.8)
    ax.set(xlabel='lag [arcsec]', ylabel=r'S$_{2}$W(l)')
    plt.tick_params(which='both', labelright=True, direction='in', right=True,  top=True)
    plt.grid(which='minor')

    return sf
##############################################################################
def sosfw2(data):

    pc=1.0

    argo=(data.I*data.RV)/data.I.sum()
    dataw= pd.DataFrame({'X': data.X, 'Y': data.Y, 'RV': argo})

    df1=dataw
    df1n=df1.to_numpy()

    xl=[row[0] for row in df1n]
    yl=[row[1] for row in df1n]
    fl=[row[2] for row in df1n]

    x1=[xl]
    y1=[yl]
    f1=[fl]

    x=list(map(list, zip(*x1)))
    y=list(map(list, zip(*y1)))
    f=list(map(list, zip(*f1)))

    sig2=2#Change to header third column
    fm=np.nanmean(f)
    fv=np.nanvar(f,ddof=1)
    fs=np.nanstd(f,ddof=1)
    l=len(f)
    S=[[fm],[fv],[fs],[l]]

    a=[[0]*(l) for i in range(l)]
    b=[[0]*(l) for i in range(l)]
    c=[[0]*(2) for i in range(l*l)]

    #normalized Second Order Structure Function Matrix [LxL]
    for i in range(l):
        for j in range(l):
          if i > j:
              a[i][j]=(((f[j][0]-fm)-(f[i][0]-fm))**2)/(fv)

    #Coord Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                b[i][j]=((x[i][0]-x[j][0])**2+(y[i][0]-y[j][0])**2)**0.5

    #LxL to Lx2
    #Main Matrix  [(L*L)x2]
    for j in range (l):
        for i in range (l):
                c[i+j*(l)][1]=a[i][j]

    for j in range (l):
        for i in range (l):
                c[i+j*(l)][0]=b[i][j]

    #Filter: Greater than "0"
    d=np.asarray(c)
    ind=np.squeeze(d[:,1])>0.0
    e=d[ind]
    #Pandas Stuff
    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Main Filter Group
    m=df.lag[1]-df.lag[0]#Lag step
    n=int(df.lag.max()//m)-1#Number of points
    #Grouping points
    dfx=dict()
    for i in range(n):
        p=0+i*(m+0.001)#rs
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    lgp=dict()
    lpc=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        lpc[i]=lgp[i]*pc
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i]-1)
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i]-1)


        g=[[0]*(8) for i in range(n)]

    for i in range(n):
        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=lpc[i]
        g[i][6]=dl[i]
        g[i][7]=derx[i]

    mx=[row[0] for row in g]
    nx=[row[1] for row in g]
    o=[row[2] for row in g]
    mpc=[row[5] for row in g]

    #Exponent

    xa=[mx]
    xe=list(map(list, zip(*xa)))
    ya=[nx]
    ye=list(map(list, zip(*ya)))

    L=len(xe)

    me=[[0]*(1) for i in range(L)]
    ne=[[0]*(1) for i in range(L)]
    oe=[[0]*(1) for i in range(L)]
    pe=[[0]*(1) for i in range(L)]
    qe=[[0]*(1) for i in range(L)]
    re=[[0]*(1) for i in range(L)]
    se=[[0]*(1) for i in range(L)]
    te=[[0]*(1) for i in range(L)]
    ue=[[0]*(1) for i in range(L)]
    we=[[0]*(1) for i in range(L)]

    for i in range (L):
        me[i][0]=math.log10(xe[i][0])
        ne[i][0]=math.log10(ye[i][0])
        oe[i][0]=(me[i][0]*ne[i][0])
        pe[i][0]=(me[i][0])**2

    def sumColumn(a):
        return [sum(col) for col in zip(*a)]

    for i in range (L+1):
        if i>0:
            qe[i-1]=sumColumn(me[:i])
            re[i-1]=sumColumn(ne[:i])
            se[i-1]=sumColumn(oe[:i])
            te[i-1]=sumColumn(pe[:i])
            ue[i-1][0]=len(x[:i])
        else:
            we[0][0]=0

    for i in range (L):
        if i>0:
            we[i][0]=((qe[i][0]*re[i][0])-(ue[i][0]*se[i][0]))/((qe[i][0])**2-ue[i][0]*te[i][0])
        else:
            we[0][0]=0

    A=[xe,ye,me,ne,oe,pe,qe,re,se,te,ue,we]
    dfe=pd.DataFrame(A)
    dfet=dfe.transpose()
    dfet.columns=['Lag.x','Nmqvd.y','Logx','Logy','Logx*Logy','Logx^2','SumLogx','SumLogy','Sum','Sum','N','Exp']

    exp=we
    ft=np.append(g,we, axis=1)
    sf=pd.DataFrame(ft)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','Lag pc','# P','ErrX','Exp']

    #Error Slope
    L=len(sf)
    m1=[[0]*(1) for i in range(L)]
    m2=[[0]*(1) for i in range(L)]
    ap=[[0]*(1) for i in range(L)]
    ee=[[0]*(1) for i in range(L)]

    for n in range(0,L):

        a=sf.Nmqvd[n]+sf.ErrY[n]
        b=sf.Nmqvd[0]-sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m1[n]=g

        a=sf.Nmqvd[n]-sf.ErrY[n]
        b=sf.Nmqvd[0]+sf.ErrY[0]
        c=sf.Lag[n]
        d=sf.Lag[0]
        e=np.log(a/b)
        f=np.log(c/d)
        g=e/f
        m2[n]=g

        ap0=[np.array(m1[n]),np.array(m2[n]),sf.Exp[n]]
        ap[n]=np.mean(ap0)

        ee0=[m1[n],m2[n],sf.Exp[n]]
        ee[n]=np.std(ee0)/np.sqrt(len(ee0))

    sf['Exp1']=m1
    sf['Exp2']=m2
    sf['Alpha']=ap
    sf['ErrT']=ee
    sf.round(3)

    fig, ax=plt.subplots()
    ax.errorbar(sf['Lag'],sf['Nmqvd'],xerr=sf.ErrX,yerr=sf.ErrY, fmt="o",color='b', ecolor='k', alpha=0.45, markersize=7.5)
    ax.set(xscale='log',yscale='log')
    ax.axhline(y=sig2, color='k', linestyle='--')
    ax.set(xlabel='lag [arcsec]', ylabel=r'S$_{2}$(l)W$_{II}$ [-]')
    plt.tick_params(which='both', labelright=True, direction='in', right=True,  top=True)
    plt.grid(which='minor')

    return sf

###############################################################################
def acf(data):

    pc=1.0

    df1=data
    df1n=df1.to_numpy()

    xl=[row[0] for row in df1n]
    yl=[row[1] for row in df1n]
    fl=[row[2] for row in df1n]

    x1=[xl]
    y1=[yl]
    f1=[fl]

    x=list(map(list, zip(*x1)))
    y=list(map(list, zip(*y1)))
    f=list(map(list, zip(*f1)))

    sig2=2*(data.RV.var())#Change to header
    fm=np.nanmean(f)
    fv=np.nanvar(f,ddof=1)
    fs=np.nanstd(f,ddof=1)
    l=len(f)
    S=[[fm],[fv],[fs],[l]]

    f=f-np.nanmean(f)

    a=[[0]*(l) for i in range(l)]
    b=[[0]*(l) for i in range(l)]
    c=[[0]*(2) for i in range(l*l)]

    #Normalized autocorrelation Function Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                a[i][j]=(f[j][0]*f[i][0])/fv

    #Coord Matrix [LxL]
    for i in range(l):
        for j in range(l):
            if i > j:
                b[i][j]=((x[i][0]-x[j][0])**2+(y[i][0]-y[j][0])**2)**0.5

    #Main Matrix  [(L*L)x2]
    for j in range (l):
        for i in range (l):
            c[i+j*(l)][1]=a[i][j]

    for j in range (l):
        for i in range (l):
            c[i+j*(l)][0]=b[i][j]

    #Filter: Greater than "0"
    d=np.asarray(c)
    ind=np.squeeze(d[:,0])>0
    e=d[ind]
    #Pandas Stuff
    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Main Filter
    m=df.lag[0]#Lag step
    #m=2*m
    n=int(df.lag.max()//m)-1#Number of points
    #Grouping points
    dfx=dict()
    for i in range(n):
        p=0+i*(m+0.001)
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    lgp=dict()
    lpc=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    #Main Matrices Statistical Properties
    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        lpc[i]=lgp[i]*pc
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/(dl[i])**0.5
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/(dl[i])**0.5


    g=[[0]*(8) for i in range(n)]

    for i in range(n):

        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=derx[i]
        g[i][4]=dfs[i]
        g[i][5]=dfv[i]
        g[i][6]=lpc[i]
        g[i][7]=dl[i]

    af=pd.DataFrame(g)
    af.columns=['Lag','Nmqvd','ErrY','ErrX','StD','Var','Lag pc','# P']
    af.loc[-1]=[0,1,0,0,0,0,0,1]
    af.index=af.index+1
    af.sort_index(inplace=True)

    fig, ax=plt.subplots()
    ax.errorbar(af['Lag'],af['Nmqvd'],xerr=af.ErrX,yerr=af.ErrY, fmt="o",color='b', ecolor='k', alpha=0.45, markersize=7.5)
    ax.axhline(y=0, color='k', linestyle='--')
    ax.set(xlabel='lag [arcsec]', ylabel=r'R(l) [-]')
    plt.tick_params(which='both', labelright=True, direction='in', right=True,  top=True)
    plt.grid(which='minor')

    return af
###############################################################################
def ps(data):

    data_2=(data.round(2)).pivot(index='Y', columns='X', values='RV')

    pc=1

    # Return multidimensional discrete Fourier transform.
    FT2a=scipy.fftpack.fft2(data_2)
    power_s0=np.abs(FT2a)
    FT2 = scipy.fftpack.fftshift(FT2a)
    power_s = np.abs(FT2)**2

    # #Distance to components
    # Create an array of "rings" parting from the center of the image.
    h  = power_s.shape[0]
    w  = power_s.shape[1]
    wc = w//2
    hc = h//2

    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)
    rdf=pd.DataFrame(r)

    # Wavenumber
    k0=[0*(1) for i in range(wc-1)]
    a=1.34125# pixel value obs data
    k0[0]=(data.X.max()-data.X.min())/2
    for i in range(wc-2):
        k0[i+1]=k0[i]-a
    np.array(k0)

    # Considering the "ring" distribution above, the last array will show the distance from the center ring (r=1,21.45 arcsec since r = 0 is not consider) and outer ring (r=16, 2.67 arcsec)
    k0=np.array(k0)
    k=k0**-1
    # We have applied the wave number definition k=1/L

    #  OBB mean
    psdf=pd.DataFrame(power_s)
    data_ps=dict()

    for i in range(wc+1):
        m=(rdf==i)
        psm=psdf[m]
        s=(psm.mean()).mean()
        err=(psm.std()/np.sqrt(len(psm))).mean()
        data_ps[i]=[i,s,err]

    psd1D_2mx=pd.DataFrame(data_ps).T

    a=psd1D_2mx[1][2:17]
    b=psd1D_2mx[2][2:17]

    psx=[k,a,b]
    psd1D_2m=pd.DataFrame(psx).T
    psd1D_2m.columns =['k', 'Pk','Err']

    fig, ax=plt.subplots()
    #sgrid = np.logspace(-1.34, -0.45)
    #plt.plot(sgrid, (10**0.7)*sgrid**(-5/3), color="k", lw=1, label="Kolmogorov", linestyle='dashed')
    #plt.plot(sgrid, (10**1.2)*sgrid**(-0.9), color="b", lw=1, label="Power law fit", linestyle='dashed')

    ax.errorbar(psd1D_2m.k,psd1D_2m.Pk, yerr=psd1D_2m.Err, fmt="o-", ecolor='k', alpha=0.5)
    ax.set(xscale='log',yscale='log')
    ax.grid()
    ax.set(xlabel='wavenumber (k), 1/arcsec', ylabel=r'$ \mathrm{P(k)}$')

    return psd1D_2m
