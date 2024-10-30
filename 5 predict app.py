# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import cross_validate,LeaveOneOut,cross_val_predict
import matplotlib.pyplot as plt
import csv
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import copy
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
def SlopeGet(Slope,m,data):
    print("正在生成斜率特征")
    slope = Slope.iloc[m].values.tolist()
    if m >= 12 and m <= 15:
        TiNum = data.iloc[:, 0] + data.iloc[:, 9] + data.iloc[:, 10]
        NiNum = data.iloc[:, 1] + data.iloc[:, 2] + data.iloc[:, 3] + data.iloc[:, 4] + data.iloc[:, 5] + data.iloc[:, 6] + data.iloc[:, 7]
        data1 = copy.deepcopy(data)
        TiNum = np.array(TiNum)
        TiNum = np.maximum((TiNum - 50), 0)
        NiNum = np.array(NiNum)
        NiNum = np.maximum((NiNum - 50), 0)
        data1.iloc[:, 0] = TiNum
        data1.iloc[:, 1] = NiNum
        new_slope1 = data1.dot(slope)
        m = data.sum(axis=1)
        new_slope1 = new_slope1 / m
    else:
        new_slope1 = data.dot(slope)
        m = data.sum(axis=1)
        new_slope1 = new_slope1 / m
    return new_slope1

#特征数据集中添加数据,data_Feature为原子信息特征集合，data为只含原子百分比的信息集合
def featureSet(data_Feature,data):
    print('正在生成非斜率特征')
    OldFeature = ['mr','ar_c','en','anum','ven','ea','dor','mass','volume','energy1','CE','EBE','YM','cs','Tm']
    data2 = pd.DataFrame()#用来存放虚拟成分的特征集合
    for i in range(data_Feature.shape[0]):
        feature = data_Feature.iloc[i,:].values.tolist()
        #feature.append(0)
        Feature = data.dot(feature)
        #部分表格的数据为百分比数据，需除100
        m = data.sum(axis=1)
        Feature = Feature / m
        data2.insert(i, OldFeature[i], Feature)
    # 增加价电子浓度特征
    cv = data2['ven'] / data2['anum']
    data2.insert(data2.shape[1], 'cv', cv)
    return data2

def RidgeResample(x,y):
    x_Train, x_verify, y_Train, y_verify = train_test_split(x, y, test_size=0.2, random_state=10)
    # 确定最优超参
    rcv = RidgeCV([i for i in np.arange(0.01,1,0.01)])
    rcv.fit(x_verify,y_verify)
    mse_FIT, mse_PRE, mse_fit_min, mse_pre_min = [], [], 10 ** 6, 10 ** 6
    #分割100次测试集和训练集
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(x_Train, y_Train, test_size=0.25, random_state=i)
        reg = Ridge(alpha=rcv.alpha_).fit(x_train, y_train)
        y_predict = reg.predict(x_test)
        y_fitted = reg.predict(x_train)
        mse_fit = mean_squared_error(y_train, y_fitted)
        mse_FIT.append(mse_fit)
        mse_pre = mean_squared_error(y_test, y_predict)
        mse_PRE.append(mse_pre)
        global results
        if mse_pre<mse_pre_min:
            mse_pre_min=mse_pre
            y_predict_min,y_fitted_min,y_train_min,y_test_min=y_predict,y_fitted,y_train,y_test
            results={'reg':reg,
                'y_predict':y_predict_min,'y_fitted':y_fitted_min,'y_train':y_train_min,'y_test':y_test_min,'mse_FIT':mse_FIT,'mse_PRE':mse_PRE}
    return results

# data_train为已经得到的训练数据,data_predict为待预测的成分集合,data_predict_feature为为待预测的成分特征集合
def GetPredictData(model,feature,data_train,data_predict,data_predict_feature):
    print('正在预测%s性能'%feature[1])
    x = data_train.loc[:, feature[0]]
    xx = data_predict_feature.loc[:, feature[0]]
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaler = scaler.transform(x)
    y = data_train.loc[:, feature[1]]
    reg_hp = model.fit(x_scaler, y)
    xx_scaler = scaler.transform(xx)
    predict_feature = reg_hp.predict(xx_scaler)
    # y = data_train.loc[:, feature[1]]
    # reg_hp = model.fit(x, y)
    # predict_feature = reg_hp.predict(xx)
    data_predict1 = copy.deepcopy(data_predict)
    data_predict1.insert(data_predict1.shape[1],feature[1],predict_feature)
    return data_predict1




# path为文件路径, order为需求的文件列名排序, targetorder为需要的成分组合排序,target为目标性能
def GetHavedData(path,order,targetorder,target):
    data= pd.read_csv(filepath_or_buffer=path)
    order1 = copy.deepcopy(order)
    order1.append(target)
    if path=='H:\\licheng\\slope-fitting\\hp\\SMA.data.training2.csv':
        data = pd.read_csv(filepath_or_buffer=path, usecols=[i for i in range(1, 13)])
        order1 = copy.deepcopy(order)
        order1.append(target)
        data.columns = order1
        # 删除hp中的TiPdCr成分
        data = data.loc[data['Ni'] != 0].copy()
        data = data.reset_index(drop=True)
    else:
        data = data[order1]
    targetpass = [y for y in order if y not in targetorder]
    for i in targetpass:
        data = data[data[i]==0].copy()
    return data
def GetModel(data,feature,):
    X = data.loc[:, feature[0]]
    X_scaler = StandardScaler().fit_transform(X)
    y = data.loc[:, feature[1]]
    params_poly = {'kernel': ['poly'], 'alpha': [i for i in np.arange(0.01, 10, 0.1)],
                   'gamma': [i for i in np.arange(0.1, 2, 0.1)], 'coef0': [0.01, 0.1, 1, 100],
                   'degree': [1, 2, 3]}
    bestscore = -10000
    for m in range(10):
        krr_poly = RandomizedSearchCV(KernelRidge(), params_poly, cv=10, verbose=0,
                                      scoring='neg_mean_squared_error', n_iter=30,
                                      random_state=m)  # cv=10即10折交叉验证
        krr_poly.fit(X_scaler,y)  # 对给定数据集选取最佳参数
        if krr_poly.best_score_ > bestscore:
            bestscore = krr_poly.best_score_
            model = krr_poly.best_estimator_
    return model


if __name__ == '__main__':

    order = ['Ti', 'Ni', 'Cu', 'Fe', 'Pd', 'Co', 'Mn', 'Cr', 'Nb', 'Zr', 'Hf']
    path_feature = 'NormalAlloyFeature.csv'
    data_feature = pd.read_csv(filepath_or_buffer=path_feature)
    data_feature = data_feature[order] # 调整data_feature列排布顺序
    path_hp = 'H:\\licheng\\slope-fitting\\hp\\SMA.data.training2.csv'
    path_enthalpy = 'H:\\licheng\\slope-fitting\\enthalpy\\data-enthalpy.csv'
    path_hysteresis = 'H:\\licheng\\slope-fitting\\hysteresis\\data-hysteresis.csv'

    data_train_hp = pd.read_pickle('hp_model')
    data_train_enthalpy = pd.read_pickle('enthalpy_model')
    data_train_hysteresis = pd.read_pickle('hysteresis_model')
    Slope = pd.read_pickle('Slope')

    data_hp = GetHavedData(path_hp, order, order, 'hp')
    data_hp2 = data_hp.drop(columns='hp')
    data_train_hp2 = featureSet(data_Feature=data_feature,data=data_hp2)
    newslope = SlopeGet(Slope=Slope,m=14,data=data_hp2)
    data_train_hp2.insert(data_train_hp2.shape[1],'slope_imp',newslope)
    data_train_hp2.insert(data_train_hp2.shape[1],'hp',data_hp.loc[:,'hp'])

    '''
    导入待预测空间并计算特征
    '''
    virtualdataTi_path='predict TiNi.csv'

    virtualdataTi_path = 'H:/licheng/slope-fitting/data/TNCH-1st.csv'

    data_space1 = pd.read_csv(filepath_or_buffer=virtualdataTi_path)

    def processing(data_space1):
        # 找到不在 order 中的列
        missing_in_order = [col for col in data_space1.columns if col not in order]
        if missing_in_order != []:
            # 提取这些列为 0 的行
            rows_with_zeros = data_space1[missing_in_order].eq(0).any(axis=1)
            # 生成新的 DataFrame
            data_space1 = data_space1[rows_with_zeros]

        # 插入缺失的列，填充为 0
        for col in order:
            if col not in data_space1.columns:
                data_space1[col] = 0

        # 确保列的顺序
        data_space1 = data_space1[order]
        return data_space1

    data_space = processing(data_space1)

    data_space_feature = featureSet(data_Feature=data_feature,data=data_space)
    newslope = SlopeGet(Slope=Slope,m=14,data=data_space)
    data_space_feature.insert(data_space_feature.shape[1],'slope_imp',newslope)


    """
    预测数据并导出结果
    """
    model_hp_feature = [('slope_imp', 'ar_c', 'ea'), 'hp']
    model_hp = GetModel(data_train_hp2, model_hp_feature)
    data_predict_hp = GetPredictData(model=model_hp,feature=model_hp_feature,data_train=data_train_hp2,data_predict=data_space,data_predict_feature=data_space_feature)
    model_enthalpy_feature = [('slope_imp', 'cv', 'YM'),'enthalpy']
    model_enthalpy = GetModel(data_train_enthalpy, model_enthalpy_feature)
    data_predict_enthalpy = GetPredictData(model=model_enthalpy,feature=model_enthalpy_feature,data_train=data_train_enthalpy,data_predict=data_space,data_predict_feature=data_space_feature)
    model_hysteresis_feature = [('slope_imp','energy1'),'hysteresis']
    model_hysteresis = GetModel(data_train_hysteresis, model_hysteresis_feature)
    data_predict_hysteresis = GetPredictData(model=model_hysteresis,feature=model_hysteresis_feature,data_train=data_train_hysteresis,data_predict=data_space,data_predict_feature=data_space_feature)
    data_predict_hp['enthalpy'] = data_predict_enthalpy.iloc[:,-1]
    data_predict_hp['hysteresis'] = data_predict_hysteresis.iloc[:, -1]
    data_predict_hp.to_csv('H:/licheng/slope-fitting/data/TNCH-1st2.csv')


