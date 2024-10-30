"""
获取enthalpy和hysteresis的特征组合和更新斜率的Dataframe
命名规则：
特征组合hp_tini、enthalpy_tini、hysteresis_tini
斜率组合hp_slope_tini、enthalpy_slope_tini、hysteresis_slope_tini
tini指数据集内部成分都为2元以上合金
删除了hysteresis中的离群点，数据命名为hysteresis_tini2、hysteresis_slope_tini2
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import copy

# 特征数据集中添加数据
def featureSet(data_Feature,data_hp,m):
    OldFeature = ['mr','ar_c','en','anum','ven','ea','dor','mass','volume','energy1','CE','EBE','YM','cs','Tm']
    for i in range(len(data_feature)):
        feature = data_feature.iloc[i].values.tolist()
        feature.append(0)
        Feature = data_hp.dot(feature)
        # 部分表格的数据为百分比数据，需除100
        Feature = Feature / m
        data_Feature.insert(i+2, OldFeature[i], Feature.values)
    return data_Feature



if __name__ == '__main__':
    path_hp = 'E:\\licheng\\licheng\\slope-fitting\\hp\\SMA.data.training.csv'
    path_enthalpy = 'E:\\licheng\\licheng\\slope-fitting\\enthalpy\\data-enthalpy.csv'
    path_feature = 'E:\\licheng\\licheng\\slope-fitting\\hp\\NormalAlloyFeature.csv'
    path_hysteresis = 'E:\\licheng\\licheng\\slope-fitting\\hysteresis\\data-hysteresis.csv'
    path_hysteresis = 'E:\\licheng\\licheng\\slope-fitting\\hysteresis\\data-hysteresis.csv'
    data_hp = pd.read_csv(filepath_or_buffer=path_hp, usecols=[i for i in range(1, 13)])
    data_enthalpy = pd.read_csv(filepath_or_buffer=path_enthalpy)
    data_hysteresis = pd.read_csv(filepath_or_buffer=path_hysteresis)
    data_feature = pd.read_csv(filepath_or_buffer=path_feature)  # 未添加斜率特征的集合
    data_Feature1, data_Feature2, data_Feature3, data_Feature4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # 添加斜率特征后的特征集合
    # 调整enthalpy和hp数据集元素顺序一致
    order = ['Ti', 'Ni', 'Cu', 'Fe', 'Pd', 'Co', 'Mn', 'Cr', 'Nb', 'Zr', 'Hf', 'enthalpy']
    data_enthalpy = data_enthalpy[order]
    # 调整feature和hp数据集元素顺序一致
    order = ['Ti', 'Ni', 'Cu', 'Fe', 'Pd', 'Co', 'Mn', 'Cr', 'Nb', 'Zr', 'Hf']
    data_feature = data_feature[order]
    # 调整hysteresis与hp数据集元素顺序一致
    order = ['Ti', 'Ni', 'Cu', 'Fe', 'Pd', 'Co', 'Mn', 'Cr', 'Nb', 'Zr', 'Hf','hysteresis']
    data_hysteresis = data_hysteresis[order]
    # 调用1 slope-get.py中获得的Slope数据集
    Slope = pd.read_pickle('Slope')
    # 组合第1代斜率和其余特征的数据集
    slope = Slope.iloc[0].values.tolist()
    slope.append(0)
    # 删除hp中的TiPdCr成分
    data_hp = data_hp.loc[data_hp['ni'] != 0]
    # 按照hp数据集摩尔平均分数构建不同组合SMA特征符
    new_feature1 = data_hp.dot(slope)
    new_feature1 = new_feature1 / 100
    data_Feature1.insert(0, 'hp', data_hp['hp'].values)
    data_Feature1.insert(1, 'slope1', new_feature1.values)
    hp_slope = copy.deepcopy(data_Feature1)
    featureSet(data_Feature1, data_hp, 100)
    # 增加价电子浓度特征
    cv = data_Feature1['ven'] / data_Feature1['anum']
    data_Feature1.insert(data_Feature1.shape[1], 'cv', cv)
    # 第1代斜率和其余特征的组合
    data_Feature1.to_pickle('hp_tini')
    # 按照enthalpy数据集摩尔平均分数构建不同组合SMA特征符
    new_feature2 = data_enthalpy.dot(slope)
    new_feature2 = new_feature2 / 100
    data_Feature2.insert(0, 'enthalpy', data_enthalpy['enthalpy'].values)
    data_Feature2.insert(1, 'slope1', new_feature2)
    enthalpy_slope = copy.deepcopy(data_Feature2)
    featureSet(data_Feature2, data_enthalpy, 100)
    # 增加价电子浓度特征
    cv = data_Feature2['ven'] / data_Feature2['anum']
    data_Feature2.insert(data_Feature2.shape[1], 'cv', cv)
    # 第1代斜率和其余特征的组合
    data_Feature2.to_pickle('enthalpy_tini')
    # 按照hysteresis数据集摩尔平均分数构建不同组合SMA特征符
    new_feature3 = data_hysteresis.dot(slope)
    new_feature3 = new_feature3 / 100
    data_Feature3.insert(0, 'hysteresis', data_hysteresis['hysteresis'].values)
    data_Feature3.insert(1, 'slope1', new_feature3)
    hysteresis_slope = copy.deepcopy(data_Feature3)
    featureSet(data_Feature3, data_hysteresis, 100)
    # 增加价电子浓度特征
    cv = data_Feature3['ven'] / data_Feature3['anum']
    data_Feature3.insert(data_Feature3.shape[1], 'cv', cv)
    # 第1代斜率和其余特征的组合
    data_Feature3.to_pickle('hysteresis_tini')
    #删除hysteresis中的异常值
    data_Feature4 = data_Feature3.drop(labels=0,axis=0)
    # 第1代斜率和其余特征的组合
    data_Feature4.to_pickle('hysteresis_tini2')
    # 组合enthalpy和hysteresis的更新斜率的Dataframe
    for m in range(1,len(Slope)):
        slope = Slope.iloc[m].values.tolist()
        slope.append(0)
        if m >= 12 and m <= 16:
            n = 0
            for data in (data_hp,data_enthalpy,data_hysteresis):
                TiNum, NiNum = [], []
                for i in range(data.shape[0]):
                    Tinum = round(data.iloc[i, 0] + data.iloc[i, 9] + data.iloc[i, 10] + data.iloc[i,8]/2, 2)
                    Ninum = round(data.iloc[i, 1] + data.iloc[i, 2] + data.iloc[i, 3] + data.iloc[i, 4] +
                                  data.iloc[i, 5] + data.iloc[i, 6] + data.iloc[i, 7] + data.iloc[i,8]/2, 2)
                    TiNum.append(Tinum)
                    NiNum.append(Ninum)
                data1 = copy.deepcopy(data)
                TiNum = np.array(TiNum)
                TiNum = np.maximum((TiNum - 50), 0)
                NiNum = np.array(NiNum)
                NiNum = np.maximum((NiNum - 50), 0)
                data1.iloc[:, 0] = TiNum
                data1.iloc[:, 1] = NiNum
                new_slope1 = data1.dot(slope)
                new_slope1 = new_slope1 / 100
                if n == 0:
                    hp_slope.insert(m + 1, 'slope%d' % (m + 1), new_slope1.values)
                if n == 1:
                    enthalpy_slope.insert(m + 1, 'slope%d' % (m + 1), new_slope1.values)
                if n == 2:
                    hysteresis_slope.insert(m + 1, 'slope%d' % (m + 1), new_slope1.values)
                n = n+1
        # 将每代斜率特征和hp放到同一张数据表
        else:
            new_slope1 = data_hp.dot(slope)
            new_slope1 = new_slope1 / 100
            hp_slope.insert(m + 1, 'slope%d' % (m+1), new_slope1.values)
            # 将每代斜率特征和enthalpy放到同一张数据表
            new_slope2 = data_enthalpy.dot(slope)
            new_slope2 = new_slope2 / 100
            enthalpy_slope.insert(m + 1, 'slope%d' % (m+1), new_slope2.values)
            # 将每代斜率特征和enthalpy放到同一张数据表
            new_slope3 = data_hysteresis.dot(slope)
            new_slope3 = new_slope3 / 100
            hysteresis_slope.insert(m + 1, 'slope%d' % (m+1), new_slope3.values)
    # 每一代更新斜率后，斜率与hp等特征的关系
    hp_slope.to_pickle('hp_slope_tini')
    enthalpy_slope.to_pickle('enthalpy_slope_tini')
    hysteresis_slope.to_pickle('hysteresis_slope_tini')
    hysteresis_slope2 = hysteresis_slope.drop(labels=0,axis=0)
    hysteresis_slope2.to_pickle('hysteresis_slope_tini2')
    # data_enthalpy1 = copy.deepcopy(data_enthalpy)
    # for i in range(data_enthalpy1.shape[1]-1):
    #     data_enthalpy1.iloc[:,i] = data_enthalpy1.iloc[:,i]*100
    # data_enthalpy1.to_csv('data-enthalpy.csv')