import numpy as np
import pandas as pd
from densratio import densratio
from skgstat import Variogram
from scipy.spatial import distance

# calculate importance weights (i.e., density ratios)
def calc_dr(tr,te,features):
    X_tr = tr[features].to_numpy()
    X_te = te[features].to_numpy()
    np.random.seed(42)
    densratio_obj = densratio(X_te, X_tr, alpha=0,sigma_range=[0.1,0.5,1,2,3,4], lambda_range=[0.1,0.3,0.5,0.8,1],kernel_num = 50)
    w = densratio_obj.compute_density_ratio(X_tr)
    return w

# calculate buffer size 
def calc_r(tr,features):
    coords = tr[['Longitude','Latitude']]
    rs=[]
    for v in features:
        V = Variogram(coords, tr[v], maxlag='median',bin_func='scott',model='matern')
        rs.append(V.describe()['effective_range']/2)
    r = round(max(rs), 2)
    return r

# get the removed id in buffer area for each fold
def rmpoints(data,k,r):
    data_cv = data.copy()
    rmpt = {}
    for i in range(1,k+1):
        rm = []
        tr_fd = data_cv.index[data_cv['fold'] != i].tolist()
        val_fd = data_cv.index[data_cv['fold'] == i].tolist()
        for j in tr_fd:
            p1 = [data_cv.iloc[j]['Longitude'], data_cv.iloc[j]['Latitude']]
            for p in val_fd:
                p2 = [data_cv.iloc[p]['Longitude'], data_cv.iloc[p]['Latitude']]
                d = distance.euclidean(p1, p2)
                if d <= r:
                    rm.append(j)
                    break
        rmpt[i] = rm
    return rmpt   

# fast version - only used in simulation experiments
def rmpoints_fast(data,k,r):
    data_cv = data.copy()
    rmpt = {}
    for i in range(1,k+1):
        test = data_cv[data_cv['fold'] == i]
        xmin,xmax = max(test['long'].min()-r,0),min(test['long'].max()+r,60)# spatial extent = 60
        ymin,ymax = max(test['lat'].min()-r,0),min(test['lat'].max()+r,60)
        train = data_cv[data_cv['fold'] != i]
        rm = train.query('%f <= long <= %f & %f <= lat <= %f'%(xmin,xmax,ymin,ymax))
        rmpt[i]=rm.index.tolist()
    return rmpt