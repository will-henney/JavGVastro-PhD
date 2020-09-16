import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
import pickle


def CPH(dt, n,reg,inst,lin,sam):

    color=itertools.cycle(("darkgoldenrod","orange","red","green","purple","blue","black"))
    marker=itertools.cycle((',','+','.','o','*'))

    X=dict()
    Y=dict()
    XY=dict()

    for i in range(n):
        X[i] = np.poly1d(np.polyfit(dt.X, dt.RV, i))

    RAgrid = np.linspace(dt.X.min(), dt.X.max())

    for i in range(n):
        Y[i]=dt.RV-X[i](dt.X)

    for i in range(n):
        XY[i]= pd.DataFrame({'X': dt.X, 'Y': dt.Y, 'RV':Y[i], 'I':dt.I})

    for i in range(n):
        print(X[i])

    for i in range(n):
        sns.distplot(XY[i].RV,bins=40,label='Orden '+str(i), color=next(color))
        plt.legend()
        plt.title('Muestra ' + sam + ' CH')

    plt.savefig('Imgs/ '+reg+inst+lin+sam+'H.png')

    plt.close()
    f = open('Res1/ '+reg+inst+lin+sam+'H.pkl',"wb")
    pickle.dump(XY,f)
    f.close()
    return XY

################################################################################

def CPV(dt, n,reg,inst,lin,sam):

    color=itertools.cycle(("darkgoldenrod","orange","red","green","purple","blue","black"))
    marker=itertools.cycle((',','+','.','o','*'))

    X=dict()
    Y=dict()
    XY=dict()

    for i in range(n):
        X[i] = np.poly1d(np.polyfit(dt.Y, dt.RV, i))

    Decgrid = np.linspace(dt.Y.min(), dt.Y.max())

    for i in range(n):
        Y[i]=dt.RV-X[i](dt.Y)

    for i in range(n):
        XY[i]= pd.DataFrame({'X': dt.X, 'Y': dt.Y, 'RV':Y[i], 'I':dt.I})

    for i in range(n):
        print(X[i])

    for i in range(n):
        sns.distplot(XY[i].RV,bins=40,label='Orden '+str(i), color=next(color))
        plt.legend()
        plt.title('Muestra ' + sam + ' CV')

    plt.savefig('Imgs/ '+reg+inst+lin+sam+'V.png')
    plt.close()
    f = open('Res1/ '+reg+inst+lin+sam+'V.pkl',"wb")
    pickle.dump(XY,f)
    f.close()
    return XY
