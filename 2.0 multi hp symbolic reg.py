import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
import copy
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.kernel_ridge import KernelRidge
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

def complexity(list):
    num = 0
    for i in list:
        if type(i) != float and type(i) != int:
            num = num+1
    return num
# list1,list2,list3分别为reg,len,rmse的集合,complexity为想要寻找的复杂度
def find(list1,list2,list3,complexity):
    rmse, results = 40, []
    for i in range(len(list1)):
        if list2[i] == complexity and list3[i] <= rmse:
            reg = list1[i]
            rmse = list3[i]
    results.append(reg)
    results.append(rmse)
    return results





if __name__ == '__main__':
    slope_str,slope_num = 'slope15', 14
    data_hp=pd.read_pickle('hp_tini')
    hp_slope = pd.read_pickle('hp_slope_tini')
    slope_imp = hp_slope.loc[:,slope_str]
    Slope = pd.read_pickle('Slope')
    Feature = ['slope_imp','mr','ar_c','en','anum','ven','ea','dor','mass','volume','energy1','CE','EBE','YM','cs','Tm','cv']
    data_hp.iloc[:,1] = slope_imp
    data_hp.rename(columns = {'slope1':'slope_imp'}, inplace=True)

    FuctionBase = ['add', 'sub', 'mul', 'div', 'sqrt', 'inv']
    Reg, RMSE, LEN = [], [], []
    for l in tqdm(range(1, 5)):
        Feature_inter = combinations(Feature, l)
        for k in tqdm(Feature_inter):
            # 分析hp的情况
            X = data_hp.loc[:,k]
            X_scaler = StandardScaler().fit_transform(X)
            y = data_hp.loc[:,'hp']
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
            for i in range(1,201):
                model = SymbolicRegressor(population_size=3000,parsimony_coefficient='auto',
                                          warm_start=True,function_set=FuctionBase,generations=1,
                                          feature_names=list(k),const_range=(-10,10),n_jobs=-1,random_state=i)
                model.fit(x_train,y_train)
                results = model._program
                y_pre = model.predict(x_test)
                mse = mean_squared_error(y_test,y_pre)
                rmse = mse**0.5
                Results = ['%s' % results]
                if rmse <= 40 and Results not in Reg:
                    Reg.append(Results)
                    RMSE.append(rmse)
                    LEN.append(complexity(results.program))

    SrResults = pd.DataFrame()
    SrResults.insert(0,'reg',Reg)
    SrResults.insert(1,'Len',LEN)
    SrResults.insert(2,'rsme',RMSE)
    #SrResults.to_pickle('SrResults_hp_2')





