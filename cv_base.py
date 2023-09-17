import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from math import sqrt
from utils import calc_dr,calc_r,rmpoints,rmpoints_fast

# calculate rmse
def calc_rmse(regr,tr,te,features):
    regr.fit(tr[features], tr['y'])
    y_pred = regr.predict(te[features])
    rmse = sqrt(mean_squared_error(te['y'], y_pred))
    return rmse

def kfcv_rmse(regr,data_cv,k,features):
    mse = cross_val_score(regr,data_cv[features], data_cv['y'], cv=k,scoring='neg_mean_squared_error')#default: not shuffle
    rmse = sqrt(abs(mse.mean()))
    return rmse

def blcv_rmse(regr,data,k,features):
    data_cv=data.copy()
    mse = []
    for i in range(1,k+1):
        train, test = data_cv[data_cv['fold'] != i], data_cv[data_cv['fold'] == i]
        regr.fit(train[features], train['y'])
        y_pred = regr.predict(test[features])
        mse.append(mean_squared_error(test['y'], y_pred))
    rmse = sqrt(sum(mse)/len(mse)) 
    return rmse

def bfcv_rmse(regr,data,k,features,rm_points):
    data_cv=data.copy()
    mse = []
    for i in range(1,k+1):
        train_id = set(data_cv.index[data_cv['fold'] != i])
        rm_id = set(rm_points[i])
        remain_id = list(train_id-rm_id)
        train = data_cv.iloc[remain_id]
        test = data_cv[data_cv['fold'] == i] 
        regr.fit(train[features], train['y'])
        y_pred = regr.predict(test[features])
        mse.append(mean_squared_error(test['y'], y_pred))
    rmse = sqrt(sum(mse)/len(mse)) 
    return rmse

def iwcv_rmse(regr,data,k,features,w):
    data_cv = data.copy()
    data_cv['weight'] = w
    data_cv['fold'] = [0]*data_cv.shape[0]
    fold = 1
    kf = KFold(n_splits=k)#default: not shuffle
    X = data_cv[features]
    for train_index, test_index in kf.split(X):
        data_cv.loc[test_index,'fold'] = fold
        fold += 1        
    mse = []
    for i in range(1,k+1):
        train, test = data_cv[data_cv['fold'] != i], data_cv[data_cv['fold'] == i] 
        regr.fit(train[features], train['y'])
        y_pred = regr.predict(test[features])
        mse_k = sum(((test['y']-y_pred)**2)*test['weight'])/len(y_pred) 
        mse.append(mse_k)
    rmse = sqrt(sum(mse)/len(mse))
    return rmse

def ibcv_rmse(regr,data,k,features,w,rm_points):
    data_cv = data.copy()
    data_cv['weight'] = w
    mse = []
    for i in range(1,k+1):
        train_id = set(data_cv.index[data_cv['fold'] != i])
        rm_id = set(rm_points[i])
        remain_id = list(train_id-rm_id)
        train = data_cv.iloc[remain_id]
        test = data_cv[data_cv['fold'] == i] 
        regr.fit(train[features], train['y'])
        y_pred = regr.predict(test[features])
        mse_k = sum(((test['y']-y_pred)**2)*test['weight'])/len(y_pred) 
        mse.append(mse_k)
    rmse = sqrt(sum(mse)/len(mse))
    return rmse

# calculate misclassification rate
def calc_mis(clf, tr,te,features):
    clf.fit(tr[features], tr['y'])
    y_pred = clf.predict(te[features])
    mis = abs(te['y'].to_numpy()-y_pred).sum()
    return np.array(mis)/te['y'].shape[0]

def kfcv_mis(clf,data,features,k):
    data_cv=data.copy()
    data_cv['fold'] = [0]*data_cv.shape[0]
    fold = 1
    kf = KFold(n_splits=k,shuffle=True, random_state=42)
    X = data_cv[features]
    for train_index, test_index in kf.split(X):
        data_cv.loc[test_index,'fold'] = fold
        fold += 1 
    mis = []
    for i in range(1,k+1):
        train, test = data_cv[data_cv['fold'] != i], data_cv[data_cv['fold'] == i] 
        clf.fit(train[features], train['y'])
        y_pred = clf.predict(test[features])
        mis.append(sum(abs(y_pred - test['y']))/test.shape[0])
    return sum(mis)/len(mis)

def blcv_mis(clf,data,features,k):
    data_cv=data.copy()
    mis = []
    for i in range(1,k+1):
        train, test = data_cv[data_cv['fold'] != i], data_cv[data_cv['fold'] == i] 
        clf.fit(train[features], train['y'])
        y_pred = clf.predict(test[features])
        mis.append(sum(abs(y_pred - test['y']))/test.shape[0])
    return sum(mis)/len(mis)

def bfcv_mis(clf,data,features,rm_points,k):
    data_cv=data.copy()
    mis = []
    for i in range(1,k+1):
        train_id = set(data_cv.index[data_cv['fold'] != i])
        rm_id = set(rm_points[i])
        remain_id = list(train_id-rm_id)
        train = data_cv.iloc[remain_id]
        test = data_cv[data_cv['fold'] == i] 
        clf.fit(train[features], train['y'])
        y_pred = clf.predict(test[features])
        mis.append(sum(abs(y_pred - test['y']))/test.shape[0])
    return sum(mis)/len(mis)

def iwcv_mis(clf,data,features,w,k):
    data_cv = data.copy()
    data_cv['weight'] = w
    data_cv['fold'] = [0]*data_cv.shape[0]
    fold = 1
    kf = KFold(n_splits=k,shuffle=True, random_state=42)
    X = data_cv[features]
    for train_index, test_index in kf.split(X):
        data_cv.loc[test_index,'fold'] = fold
        fold += 1 
    miss = []
    for i in range(1,k+1):
        train, test = data_cv[data_cv['fold'] != i], data_cv[data_cv['fold'] == i] 
        clf.fit(train[features], train['y'])
        y_pred = clf.predict(test[features])
        mis = abs(y_pred - test['y'])
        miss.append(sum(mis*test['weight'])/test.shape[0])
    return sum(miss)/len(miss)

def ibcv_mis(clf,data,features,r,w,rm_points,k):
    data_cv = data.copy()
    data_cv['weight'] = w
    miss = []
    for i in range(1,k+1):
        train_id = set(data_cv.index[data_cv['fold'] != i])
        rm_id = set(rm_points[i])
        remain_id = list(train_id-rm_id)
        train = data_cv.iloc[remain_id]
        test = data_cv[data_cv['fold'] == i] 
        clf.fit(train[features], train['y'])
        y_pred = clf.predict(test[features])
        mis = abs(y_pred - test['y'])
        miss.append(sum(mis*test['weight'])/test.shape[0])
    return sum(miss)/len(miss)

# compute test and cv rmses for regression task
def run_regr(features, y, k, trfile, tefile, bcvfile, outfile):
    tr = pd.read_csv(trfile).rename(columns={y: "y"})
    te = pd.read_csv(tefile).rename(columns={y: "y"})
    print("Training set:\n", tr.shape[0])
    print("Test set:\n", te.shape[0])
    r = calc_r(tr,features)
    print("Buffer size:", r)     
    w = calc_dr(tr,te,features)
    
    names = ["Linear","KRR","SVR","KNN", "RF"]
    regrs = [LinearRegression(),KernelRidge(),SVR(),KNeighborsRegressor(),RandomForestRegressor(random_state=42)]
    test_error,kfcv_error,blcv_error,bfcv_error,iwcv_error,ibcv_error = [],[],[],[],[],[]
    bcvfold = pd.read_csv(bcvfile).drop(columns=['Unnamed: 0'])
    tr_bcv = pd.concat([tr, bcvfold], axis=1, sort=False).rename(columns={"x": "fold"})
    rm_points = rmpoints(tr_bcv,k,r)
    
    for regr in regrs:
        test_error.append(calc_rmse(regr,tr,te,features))
        kfcv_error.append(kfcv_rmse(regr,tr,k,features))
        blcv_error.append(blcv_rmse(regr,tr_bcv,k,features))
        bfcv_error.append(bfcv_rmse(regr,tr_bcv,k,features,rm_points))
        iwcv_error.append(iwcv_rmse(regr,tr,k,features,w))
        ibcv_error.append(ibcv_rmse(regr,tr_bcv,k,features,w,rm_points))
    
    # write result to csv file
    df ={'regr':names,'test':test_error,'kfcv':kfcv_error,'blcv':blcv_error,'bfcv':bfcv_error,'iwcv':iwcv_error,'ibcv':ibcv_error}
    summary = pd.concat([pd.Series(v, name=k) for k, v in df.items()], axis=1)
    summary.to_csv(outfile, index = False, header=True) 

# compute test and cv misclassification rate for classification task
def run_clf(features,y, k, trfile, tefile, bcvfile, outfile):
    tr = pd.read_csv(trfile).rename(columns={y: "y"})
    te = pd.read_csv(tefile).rename(columns={y: "y"})
    print("Training set:\n", tr.shape[0])
    print("Test set:\n", te.shape[0])
    r = calc_r(tr,features)
    print("Buffer size:", r)     
    w = calc_dr(tr,te,features)
    
    names = ["Ridge","KNN","RF","LSVM","NB"]
    classifiers = [RidgeClassifier(),KNeighborsClassifier(),RandomForestClassifier(random_state=42),SVC(kernel="linear", random_state=42),GaussianNB()]
    test_error,kfcv_error,blcv_error,bfcv_error,iwcv_error,ibcv_error = [],[],[],[],[],[]
    bcvfold = pd.read_csv(bcvfile).drop(columns=['Unnamed: 0'])
    tr_bcv = pd.concat([tr, bcvfold], axis=1, sort=False).rename(columns={"x": "fold"})
    rm_points = rmpoints(tr_bcv,k,r)
    
    for clf in classifiers:
        test_error.append(calc_mis(clf,tr,te,features))
        kfcv_error.append(kfcv_mis(clf,tr,features,k))
        blcv_error.append(blcv_mis(clf,tr_bcv,features,k))
        bfcv_error.append(bfcv_mis(clf,tr_bcv,features,rm_points,k))
        iwcv_error.append(iwcv_mis(clf,tr,features,w,k))
        ibcv_error.append(ibcv_mis(clf,tr_bcv,features,r,w,rm_points,k))
    
    # write result to csv file
    df ={'clf':names,'test':test_error,'kfcv':kfcv_error,'blcv':blcv_error,'bfcv':bfcv_error,'iwcv':iwcv_error,'ibcv':ibcv_error}
    summary = pd.concat([pd.Series(v, name=k) for k, v in df.items()], axis=1)
    summary.to_csv(outfile, index = False, header=True)
         
def run_sim(scn, bcvfile, outfile): #scn: scenario
    sac_r = [1,4,8,12] # spatial autocorrelation range
    k = 9 # number of folds
    sim = 3 # number of simulations, 3 for demonstration, 100 in paper
    m = 1800 # training set size
    n = 500 # test set size
    features = ['X1','X2']
    regr = LinearRegression()
    
    for r in sac_r:
        test_error,kfcv_error,blcv_error,bfcv_error,iwcv_error,ibcv_error = [],[],[],[],[],[]
        for i in range(1,sim+1):
            # set dataset for each scenario
            if scn == 'sd':
                tr = pd.read_csv("./data/simulation/sac%d/c1_tr_%d.csv" % (r,i))
                te = pd.read_csv( "./data/simulation/sac%d/c1_te_%d.csv" % (r,i))
            elif scn == 'si':
                if i == sim: #in the last one, use the first te
                    tr = pd.read_csv("./data/simulation/sac%d/c1_tr_%d.csv" % (r,i))
                    te = pd.read_csv("./data/simulation/sac%d/c1_te_%d.csv" % (r,1))
                else: #use the next as te
                    tr = pd.read_csv("./data/simulation/sac%d/c1_tr_%d.csv" % (r,i))
                    te = pd.read_csv("./data/simulation/sac%d/c1_te_%d.csv" % (r,i+1))
            elif scn == 'sdcs':
                tr = pd.read_csv("./data/simulation/sac%d/c3_tr_%d.csv" % (r,i))
                te = pd.read_csv( "./data/simulation/sac%d/c3_te_%d.csv" % (r,i))
            elif scn == 'sics':
                tr = pd.read_csv("./data/simulation/sac%d/c1_tr_%d.csv" % (r,i))
                te = pd.read_csv( "./data/simulation/sac%d/c4_te_%d.csv" % (r,i))
            elif scn == 'sirs': #training set range is fixed to 12.
                if r==12 and i !=sim:
                    tr = pd.read_csv("./data/simulation/sac%d/c1_tr_%d.csv" % (12,i))
                    te = pd.read_csv("./data/simulation/sac%d/c1_te_%d.csv" % (r,i+1))
                elif r==12 and i ==sim:
                    tr = pd.read_csv("./data/simulation/sac%d/c1_tr_%d.csv" % (12,i))
                    te = pd.read_csv("./data/simulation/sac%d/c1_te_%d.csv" % (r,1))
                else:
                    tr = pd.read_csv("./data/simulation/sac%d/c1_tr_%d.csv" % (12,i))
                    te = pd.read_csv("./data/simulation/sac%d/c1_te_%d.csv" % (r,i))
            elif scn == 'sipcs':
                tr = pd.read_csv("./data/simulation/sac%d/c1_tr_%d.csv" % (r,i))
                te_tmp1 = pd.read_csv("./data/simulation/sac%d/c4_te_%d.csv" % (r,i))
                te_tmp2 = te_tmp1.filter(['X3','X4'], axis=1).copy()
                te = te_tmp2.rename(columns={"X3": "X1","X4": "X2"})
            else:
                print("No specific scenario!")
                
            e_tr = np.random.default_rng(seed=i).normal(0, 0.1, m)
            e_te = np.random.default_rng(seed=i).normal(0, 0.1, n)
            y_tr = tr['X1'] + tr['X2'] + tr['X1']*tr['X2'] + e_tr
            tr['y'] = y_tr
            y_te = te['X1'] + te['X2'] + te['X1']*te['X2'] + e_te
            te['y'] = y_te

            if scn == 'sirs':
                bcvfold = pd.read_csv(bcvfile % (12,i)).drop(columns=['Unnamed: 0'])
            else:
                bcvfold = pd.read_csv(bcvfile % (r,i)).drop(columns=['Unnamed: 0'])
            # here we use the same block size for blcv, bfcv and ibcv
            # need to pass different tr_bcv files for different block sizes of these cv methods
            tr_bcv = pd.concat([tr, bcvfold], axis=1, sort=False).rename(columns={"x": "fold"})
            w = calc_dr(tr,te,features)
            rm_points = rmpoints_fast(tr_bcv,k,r)
            
            test_error.append(calc_rmse(regr,tr,te,features))
            kfcv_error.append(kfcv_rmse(regr,tr,k,features)) 
            blcv_error.append(blcv_rmse(regr,tr_bcv,k,features))
            bfcv_error.append(bfcv_rmse(regr,tr_bcv,k,features,rm_points))
            iwcv_error.append(iwcv_rmse(regr,tr,k,features,w))
            ibcv_error.append(ibcv_rmse(regr,tr_bcv,k,features,w,rm_points))
        
        # write result to csv file
        d = {'test':test_error,'kfcv':kfcv_error,'blcv':blcv_error,'bfcv':bfcv_error,'iwcv':iwcv_error,'ibcv':ibcv_error}
        summary = pd.concat([pd.Series(v, name=k) for k, v in d.items()], axis=1)
        summary.to_csv(outfile % r, index = False, header=True)
    