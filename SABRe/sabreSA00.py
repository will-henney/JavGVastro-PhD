import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fftpack

############################SABRE############################################
###########Statistical Analysis of Bidimensional Regions#####################
############################################################################
#Input dataframe: ['X' 'Y' 'RV' 'I'] as columns
############################content:########################################
#sosf -Second Order Structure Function
#sosfn - Normalized Second Order Structure Function
#acf - Auto correlation Function
#sosfw- Weighted Second Order Structure Function
#sosfwp- Weighted property Second Order Structure Function
#sosfh - SF provided by Dr. Will
#pswk - TWK

##############################################################################
#################Second Order Structure Function##############################
##############################################################################
def sosf(data,m):
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

    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Structure Function Data Groups
    #m separation between lags min valu=min lag
    n=df.lag.max()//m#Number of points

    #Grouping points
    dfx=dict()

    for i in range(int(n)):
        p=0+i*m
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    #Main Matrix Statistical Properties Groups

    lgp=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    n=len(dfx)

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i])
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i])

    g=[[0]*(7) for i in range(n)]

    for i in range(n):

        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=dl[i]
        g[i][6]=derx[i]

    sf=pd.DataFrame(g)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','# P','ErrX']

    return sf

##############################################################################
################# Norm Second Order Structure Function  ######################
##############################################################################
def sosfn(data,m):
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
    #for i in range(l):
    #    for j in range(l):
    #        if i > j:
    #            a[i][j]=(((f[j][0])-(f[i][0]))**2)

    #Normalized Second Order Structure Function Matrix [LxL]
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

    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Structure Function Data Groups
    #m=1.341#separation between lags mib valu=min lag
    n=int(df.lag.max())//m#Number of points

    #Grouping points
    dfx=dict()

    for i in range(int(n)):
        p=0+i*m
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    #Main Matrix Statistical Properties Groups

    lgp=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    n=len(dfx)

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i])
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i])

    g=[[0]*(7) for i in range(n)]

    for i in range(n):

        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=dl[i]
        g[i][6]=derx[i]

    sf=pd.DataFrame(g)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','# P','ErrX']

    return sf

##############################################################################
##################### Auto Correlation Function  #############################
##############################################################################
def acf(data,m):
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

    #Autocorrelation Function Matrix [LxL]

    f=f-np.nanmean(f)

    for i in range(l):
        for j in range(l):
            if i > j:
                a[i][j]=(f[j][0]*f[i][0])/fv

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
    ind=np.squeeze(d[:,0])>0.0
    e=d[ind]

    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Structure Function Data Groups
    #m=1.341#separation between lags mib valu=min lag
    n=int(df.lag.max())//m#Number of points

    #Grouping points
    dfx=dict()

    for i in range(int(n)):
        p=0+i*m
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    #Main Matrix Statistical Properties Groups

    lgp=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    n=len(dfx)

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i])
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i])

    g=[[0]*(7) for i in range(n)]

    for i in range(n):

        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=dl[i]
        g[i][6]=derx[i]

    sf=pd.DataFrame(g)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','# P','ErrX']

    sf.loc[-1]=[0,1,0,0,0,1,0]
    sf.index=sf.index+1
    sf.sort_index(inplace=True)

    return sf
##############################################################################
############# Weighted Second Order Structure Function  ######################
##############################################################################
def sosfw(data,m):

    df1=abs(data)
    df1n=df1.to_numpy()

    df1=abs(data)
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

    sig2=2*(data.RV.var())#Change to header third column
    fm=np.nanmean(f)
    fv=np.nanvar(f,ddof=1)
    fs=np.nanstd(f,ddof=1)
    l=len(f)
    S=[[fm],[fv],[fs],[l]]

    a=[[0]*(l) for i in range(l)]
    b=[[0]*(l) for i in range(l)]
    c=[[0]*(2) for i in range(l*l)]

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

    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    n=int(df.lag.max())//m#Number of points

    #Grouping points
    dfx=dict()

    for i in range(int(n)):
        p=0+i*m
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    lgp=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    n=len(dfx)

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i])
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i])

    g=[[0]*(7) for i in range(n)]

    for i in range(n):

        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=dl[i]
        g[i][6]=derx[i]

    sf=pd.DataFrame(g)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','# P','ErrX']

    return sf
##############################################################################
############# Weighted Property S.O. Structure Function  ####################
##############################################################################
def sosfwp(data,m):

    argo=(data.I*data.RV)/data.I.sum()
    dataw= pd.DataFrame({'X': data.X, 'Y': data.Y, 'RV': argo})
    data=dataw

    df1=abs(data)
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
                #a[i][j]=(((f[j][0]-fm)-(f[i][0]-fm))**2)/(fv)

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

    df=pd.DataFrame(e)
    df.columns=['lag','qvd']#squared velocities difference

    #Structure Function Data Groups
    #m separation between lags min valu=min lag
    n=df.lag.max()//m#Number of points

    #Grouping points
    dfx=dict()

    for i in range(int(n)):
        p=0+i*m
        q=m+i*m
        dfx[i]=df[df['lag'].between(p,q)]

    #Main Matrix Statistical Properties Groups

    lgp=dict()
    dl=dict()
    dery=dict()
    dfm=dict()
    dfv=dict()
    dfs=dict()
    derx=dict()
    dlm=dict()
    dlv=dict()
    dls=dict()

    n=len(dfx)

    for i in range(n):
        lgp[i]=(dfx[i]["lag"].max())
        dl[i]=len(dfx[i])#Numer of points each group
        #errorY
        dfm[i]=dfx[i]["qvd"].mean()
        dfv[i]=dfx[i]["qvd"].var()
        dfs[i]=dfx[i]["qvd"].std()
        if dl[i]>0:
            dery[i]=dfs[i]/np.sqrt(dl[i])
        #errorX
        dlm[i]=dfx[i]["lag"].mean()
        dlv[i]=dfx[i]["lag"].var()
        dls[i]=dfx[i]["lag"].std()
        if dl[i]>0:
            derx[i]=dls[i]/np.sqrt(dl[i])

    g=[[0]*(7) for i in range(n)]

    for i in range(n):

        g[i][0]=lgp[i]
        g[i][1]=dfm[i]
        g[i][2]=dery[i]
        g[i][3]=dfs[i]
        g[i][4]=dfv[i]
        g[i][5]=dl[i]
        g[i][6]=derx[i]

    sf=pd.DataFrame(g)
    sf.columns=['Lag','Nmqvd','ErrY','StD','Var','# P','ErrX']

    return sf
##############################################################################
#################Second Order Structure Function Dr. Will Henney##############
##############################################################################
def sosfh(data,m):

    df=data
    dfn=df.to_numpy()
    df=df.rename(columns={'X': 'RAdeg','Y':'DEdeg', 'RV':'vHa'})###########!!!!

    df1 = pd.DataFrame({'RA': df.RAdeg, 'DE': df.DEdeg, 'V': df.vHa, '_key': 1})
    df2 = df1.copy()

    pairs = pd.merge(df1, df2, on='_key', suffixes=('', '_')).drop('_key', 1)
    pairs.index = pd.MultiIndex.from_product((df1.index, df2.index))

    pairs.loc[:, 'dDE'] = 1*(pairs.DE - pairs.DE_)
    pairs.loc[:, 'dRA'] = 1*(pairs.RA - pairs.RA_)*np.cos(np.radians(0.5*(pairs.DE + pairs.DE_)))
    pairs.loc[:, 's'] = np.hypot(pairs.dRA, pairs.dDE)
    pairs.loc[:, 'log_s'] = np.log10(pairs.s)
    pairs.loc[:, 'dV'] = pairs.V - pairs.V_
    pairs.loc[:, 'dV2'] = pairs.dV**2
    pairs.loc[:, 'log_dV2'] = np.log10(pairs.dV**2)
    pairs.loc[:, 'VV_mean'] = 0.5*(pairs.V + pairs.V_)

    pairs = pairs[(pairs.dDE > 0.0) & (pairs.dRA > 0.0)]

    pairs.loc[:, 's_class'] = pd.Categorical((2*pairs.log_s + 0.5).astype('int'), ordered=True)

    pairs.s_class[pairs.s_class == 0] = 1

    #for j in range(7):
    #    print()
    #    print("s_class =", j)
    #    print(pairs[pairs.s_class == j][['dV2', 'log_s']].describe())

    sig2 = pairs.dV2.mean()
    sig2a = 2*np.var(df1.V)

    ngroup = m
    groups = np.arange(len(pairs)) // ngroup
    table = pairs[['s', 'dV2']].sort_values('s').groupby(groups).describe()
    #fig, ax = plt.subplots(figsize=(7, 7))
    s = table[('s', 'mean')]
    e_s = table[('s', 'std')]
    b2 = table[('dV2', 'mean')]
    ng = table[('dV2', 'count')]
    e_b2 = table[('dV2', 'std')]/np.sqrt(ng - 1)

    table=[s,b2,e_s,e_b2]
    sf=pd.DataFrame(table)
    sf=sf.transpose()
    sf=sf.set_axis(['Lag', 'Nmqvd', 'ErrX', 'ErrY'], axis=1, inplace=False)

    return sf
##############################################################################
#################           Teorema WK                          ##############
##############################################################################

def pswk(data,data1):

    bt=np.fft.fft(data.Nmqvd)
    ct=bt*np.conj(bt)

    y=ct[0:len(ct)//2]

    x=np.linspace(data.Lag.max(),data.Lag[1],len(y))
    x=(2*np.pi)/x

    ps=[abs(x),abs(y*data1.RV.var())]
    psd=pd.DataFrame(ps).T
    psd=psd.rename(columns={0:'k',1:'Pk'})

    return psd
