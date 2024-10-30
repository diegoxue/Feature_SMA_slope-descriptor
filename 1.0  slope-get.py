"""
TixNiy，x和y值的变化，会导致不同成分的外推值不相同，即数据点并不在同一条直线上，而是几条平行线
直接拟合会增大误差，如何区分
1 以0.5为步长，贫Ni端变化不大，记为-1，等原子比记为0，富Ti记为1、2、3、4
 先只用等原子比训练模型，预测时，非等原子比再加上修正项
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import copy

#线性回归模型求斜率
def linearmodel(data_one,str,data_slope):
    x = data_one[str].values.reshape(-1,1)
    y = data_one['hp'].values.reshape(-1,1)
    reg=LinearRegression().fit(x,y)
    data_slope.extend(reg.coef_[0].tolist())
    return data_slope
#改进斜率描述，其余元素先减去TiNi斜率后，再单变量回归
def newlinearmodel(data_one,str,data_slope):
    ti,ni = reg_tini.coef_[0],reg_tini.coef_[1]
    x = data_one[str].values.reshape(-1, 1)
    y = (data_one['hp'] - ti*data_one['ti']-ni*data_one['ni']).values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    data_slope.extend(reg.coef_[0].tolist())
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(x, y, s=9)
    # ax.set_ylabel('Measured hp')  # 设置y坐标轴名称
    # ax.set_xlabel(str)  # 设置x坐标轴名称
    # ax[1].scatter(x0, y, s=9)
    # ax[1].scatter(x0, yFit, s=9)
    # ax[1].legend(labels = ['Measured hp','fitted hp'],loc='best')
    # ax[1].set_xlabel('new %s'%str)  # 设置x坐标轴名称
    plt.show()
    return data_slope

#改进斜率描述，其余元素先减去TiNi斜率后，再单变量回归
def newlinearmodel2(data_one,str,data_slope):
    #-207.49为Ren综述中富Ni端的斜率
    ti, ni = 0, -207.49
    x = data_one[str].values.reshape(-1, 1)
    y = (data_one['hp'] - ti*data_one['ti']-ni*data_one['ni']).values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    data_slope.extend(reg.coef_[0].tolist())
    return data_slope

#改进斜率描述，其余元素先减去TiNi斜率后，再单变量回归
def newlinearmodel3(data_one,str,data_slope):
    ti, ni = 0, -109.27
    x = data_one[str].values.reshape(-1, 1)
    y = (data_one['hp'] - ti*data_one['ti']-ni*data_one['ni']).values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    data_slope.extend(reg.coef_[0].tolist())
    return data_slope
#改进斜率描述，50差值模型，TiNiX先减去（Ni-50）*slopeNi后，再单变量回归
def newlinearmodel4(data_one,str,data_slope):
    ti, ni = 0, -207.49
    x = data_one[str].values.reshape(-1, 1)
    y = (data_one['hp'] - ti*data_one['TiNum']-ni*(data_one['NiNum']-50)).values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    data_slope.extend(reg.coef_[0].tolist())
    return data_slope
#改进斜率描述，50差值模型，TiNiX先减去（Ni-50）*slopeNi后，再单变量回归
def newlinearmodel4(data_one,str,data_slope):
    ti, ni = 0, -207.49
    x = data_one[str].values.reshape(-1, 1)
    y = (data_one['hp'] - ti*data_one['TiNum']-ni*(data_one['NiNum']-50)).values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    data_slope.extend(reg.coef_[0].tolist())
    return data_slope
#改进斜率描述，50差值模型，TiNiX先减去（Ni-50）*slopeNi后，slopeNi分为贫Ni和富Ni,再单变量回归
def newlinearmodel5(data_one,str,data_slope):
    ti, ni = 1.72, -207.49
    x = data_one[str].values.reshape(-1, 1)
    if str == 'nb':
        y = (data_one['hp'] - ti*np.maximum((data_one['TiNum']+data_one['nb']/2-50),0)-ni*np.maximum((data_one['NiNum']+data_one['nb']/2-50),0)).values.reshape(-1, 1)
    else:
        y = (data_one['hp'] - ti*np.maximum((data_one['TiNum']-50),0)-ni*np.maximum((data_one['NiNum']-50),0)).values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    data_slope.extend(reg.coef_[0].tolist())
    return data_slope
def newlinearmodel6(data_one,str,data_slope):
    ti, ni = 18.42, -109.27
    x = data_one[str].values.reshape(-1, 1)
    if str == 'nb':
        y = (data_one['hp'] - ti*np.maximum((data_one['TiNum']+data_one['nb']/2-50),0)-ni*np.maximum((data_one['NiNum']+data_one['nb']/2-50),0)).values.reshape(-1, 1)
    else:
        y = (data_one['hp'] - ti*np.maximum((data_one['TiNum']-50),0)-ni*np.maximum((data_one['NiNum']-50),0)).values.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    data_slope.extend(reg.coef_[0].tolist())
    return data_slope
def newlinearmodel7(data_one,str,data_slope):
    ti, ni = 18.42, -109.27
    MseFit, x0, yFit, a0, coef = 10000, [], [], 1, 0
    if str in ['hf','zr','pd']:
        sloperange = np.arange(2,10,0.1)
    else:
        sloperange = np.arange(-50,-2,0.1)
    for a in sloperange:
        if str == 'nb':
            x = (ti * np.maximum((data_one['TiNum'] + data_one['nb'] / 2 - 50), 0) + ni * np.maximum(
                (data_one['NiNum'] + data_one['nb'] / 2 - 50), 0) + a*data_one[str]).values.reshape(-1, 1)
        else:
            x = (ti * np.maximum((data_one['TiNum'] - 50), 0) + ni * np.maximum(
                (data_one['NiNum'] - 50), 0) + a*data_one[str]).values.reshape(-1, 1)
        y = data_one['hp'].values.reshape(-1,1)
        reg = LinearRegression().fit(x,y)
        y_fitted = reg.predict(x)
        mse_fit = mean_squared_error(y,y_fitted)
        # 保证修正系数a不为0
        if mse_fit <MseFit and abs(a) >= 10**-5:
            MseFit = mse_fit
            x0 = x
            yFit = y_fitted
            a0 = a
    data_slope.append(a0)
    return data_slope
#采用Ti和Ni的差值模型，其余元素分别替代Ti和Ni,a为正，替代Ni，a为负，替代Ti
def RatioModel(data_one,str,data_slope):
    MseFit, x0, yFit, a0, coef = 1000, [], [], 1, 0
    for a in np.arange(-5,5,0.1):
        x = (data_one['ni']-data_one['ti']+a*data_one[str]).values.reshape(-1,1)
        y = data_one['hp'].values.reshape(-1,1)
        reg = LinearRegression().fit(x,y)
        y_fitted = reg.predict(x)
        mse_fit = mean_squared_error(y,y_fitted)
        # 保证修正系数a不为0
        if mse_fit <MseFit and abs(a) >= 10**-5:
            MseFit = mse_fit
            x0 = x
            yFit = y_fitted
            a0 = a
            coef = reg.coef_[0]
    #print("当%s的修正系数为%.2f时，斜率为%f，mse_fit最小，其值为%f"%(str,a0,coef,MseFit))
    # X = data_one[str].values.reshape(-1,1)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].scatter(X, y, s=9)
    # ax[0].set_ylabel('Measured hp')  # 设置y坐标轴名称
    # ax[0].set_xlabel(str)  # 设置x坐标轴名称
    # ax[1].scatter(x0, y, s=9)
    # ax[1].scatter(x0, yFit, s=9)
    # ax[1].legend(labels = ['Measured hp','fitted hp'],loc='best')
    # ax[1].set_xlabel('new %s'%str)  # 设置x坐标轴名称
    # plt.show()
    data_slope.append(a0*float(coef))
    return data_slope
#特征数据集中添加数据
def featureSet(data_Feature,data_hp,m):
    OldFeature = ['mr','ar_c','en','anum','dor','mass','volume','energy1','CE','EBE','YM','cs','Tm']
    for i in range(len(data_feature)):
        feature = data_feature.iloc[i].values.tolist()
        feature.append(0)
        Feature = data_hp.dot(feature)
        #部分表格的数据为百分比数据，需除100
        Feature = Feature / m
        data_Feature.insert(i+2, OldFeature[i], Feature)
    return data_Feature

# 获得Nb1、2、3信息,分别统计斜率，只返回同时替代ti和ni的斜率
def NbSlope(Nb):
    nbslope = []
    Nb1,Nb2,Nb3 = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    # 遍历Nb中每一行,判断Ti和Ni的大小，依次存入nbslope中
    # Nb1ni大于ti，Nb2ti大于ni，Nb3ti和ni1相等
    for i in range(12):
        if Nb.iloc[i,0] > Nb.iloc[i,1]:
            Nb1 = Nb1._append(Nb.iloc[i], ignore_index=True)
        elif Nb.iloc[i,0] < Nb.iloc[i,1]:
            Nb2 = Nb2._append(Nb.iloc[i], ignore_index=True)
        else:
            Nb3 = Nb3._append(Nb.iloc[i], ignore_index=True)
    for i in (Nb1,Nb2,Nb3):
        if i.empty:
            pass
        else:
            newlinearmodel3(i,'nb',nbslope)
        # x = i['nb'].values.reshape(-1, 1)
        # y = i['hp'].values.reshape(-1, 1)
        # reg = LinearRegression().fit(x, y)
        # nbslope.extend(reg.coef_[0].tolist())
    #slope.append(nbslope[2])
    return nbslope[1]



if __name__ == '__main__':
    path_hp = 'H:\\licheng\\slope-fitting\\hp\\SMA.data.training.csv'
    path_enthalpy='H:\\licheng\\slope-fitting\\enthalpy\\traindata_enthalpy.csv'
    path_feature='H:\\licheng\\slope-fitting\\hp\\NormalAlloyFeature.csv'
    data_hp = pd.read_csv(filepath_or_buffer=path_hp, usecols=[i for i in range(1, 13)])
    data_enthalpy = pd.read_csv(filepath_or_buffer=path_enthalpy)
    data_feature = pd.read_csv(filepath_or_buffer=path_feature)#未添加斜率特征的集合
    data_Feature1,data_Feature2 = pd.DataFrame(),pd.DataFrame()#添加斜率特征后的特征集合
    #调整enthalpy和hp数据集元素顺序一致
    order=['Ti', 'Ni', 'Cu', 'Fe', 'Pd', 'Co', 'Mn', 'Cr', 'Nb', 'Zr', 'Hf', 'enthalpy']
    data_enthalpy = data_enthalpy[order]
    # 调整feature和hp数据集元素顺序一致
    order = ['Ti', 'Ni', 'Cu', 'Fe', 'Pd', 'Co', 'Mn', 'Cr', 'Nb', 'Zr', 'Hf']
    data_feature = data_feature[order]
    #增加Ti端和Ni端的信息
    TiNum,NiNum = [],[]
    for i in range(data_hp.shape[0]):
        Tinum = round(data_hp.iloc[i,0]+data_hp.iloc[i,9]+data_hp.iloc[i,10],2)
        Ninum = round(data_hp.iloc[i,1]+data_hp.iloc[i, 2]+data_hp.iloc[i,3]+data_hp.iloc[i,4]+
                      data_hp.iloc[i,5]+data_hp.iloc[i,6]+data_hp.iloc[i,7],2)
        TiNum.append(Tinum)
        NiNum.append(Ninum)
    data_hp.insert(data_hp.shape[1],'TiNum',TiNum)
    data_hp.insert(data_hp.shape[1], 'NiNum', NiNum)
    #搜索单一元素参杂数据，即TiNiX数据
    data1 = copy.deepcopy(data_hp)
    Cu, Fe, Pd, Co, Mn, Cr, Nb, Zr, Hf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(len(data_hp)):
        num0=data_hp.iloc[i].value_counts().loc[0.0]#统计每一行0的个数
        if num0 != 8:
            data1.drop(i,inplace=True)

    Ti50Ni50 = data_hp[(data_hp['ti'] == 50) & (data_hp['ni'] == 50)].copy()
    Ti505Ni495 = data_hp[(data_hp['ti'] == 50.5) & (data_hp['ni'] == 49.5)].copy()
    Ti51Ni49 = data_hp[(data_hp['ti'] == 51) & (data_hp['ni'] == 49)].copy()

    Cu = data1.loc[data_hp['cu'] != 0].copy()
    Cu1 = Cu._append(Ti50Ni50, ignore_index=True)
    Cu50 = Cu[Cu['ti'] == 50].copy()
    Cu50.sort_values(by='cu',inplace=True,ascending=True)
    Cu = Cu._append(Ti50Ni50, ignore_index=True)

    Fe = data1.loc[data_hp['fe'] != 0].copy()
    Fe1 = Fe._append(Ti50Ni50, ignore_index=True)
    Fe50 = Fe[Fe['ti'] == 50].copy()
    Fe50.sort_values(by='fe', inplace=True, ascending=True)
    Fe = Fe._append(Ti50Ni50, ignore_index=True)

    Pd = data1.loc[(data_hp['pd'] != 0) & (data_hp['ni'] != 0)].copy()
    Pd1 = Pd._append(Ti50Ni50, ignore_index=True)
    Pd50 = Pd[Pd['ti'] == 50].copy()
    Pd50.sort_values(by='pd', inplace=True, ascending=True)
    Pd = Pd._append(Ti50Ni50, ignore_index=True)
    Co = data1.loc[data_hp['co'] != 0].copy()
    Co1 = Co._append(Ti50Ni50, ignore_index=True)
    Co50 = Co[Co['ti'] == 50].copy()
    Co50.sort_values(by='co', inplace=True, ascending=True)
    Co = Co._append(Ti50Ni50, ignore_index=True)
    Mn = data1.loc[data_hp['mn'] != 0].copy()
    Mn1 = Mn._append(Ti50Ni50, ignore_index=True)
    Mn50 = Mn[Mn['ti'] == 50].copy()
    Mn50.sort_values(by='mn', inplace=True, ascending=True)
    Mn = Mn._append(Ti50Ni50, ignore_index=True)
    Cr = data1.loc[(data_hp['cr'] != 0) & (data_hp['ni'] != 0)].copy()
    Cr1 = Cr._append(Ti50Ni50, ignore_index=True)
    Cr50 = Cr[Cr['ti'] == 50].copy()
    Cr50.sort_values(by='cr', inplace=True, ascending=True)
    Cr = Cr._append(Ti50Ni50, ignore_index=True)
    Nb = data1.loc[data_hp['nb'] != 0].copy()
    Nb1 = Nb._append(Ti50Ni50, ignore_index=True)
    Nb50 = Nb[Nb['ti'] == Nb['ni']].copy()
    Nb50.sort_values(by='nb', inplace=True, ascending=True)
    Nb = Nb._append(Ti50Ni50, ignore_index=True)
    Zr = data1.loc[data_hp['zr'] != 0].copy()
    Zr1 = Zr._append(Ti505Ni495, ignore_index=True)
    Zr495 = Zr[Zr['ni'] == 49.5].copy()
    Zr495.sort_values(by='zr', inplace=True, ascending=True)
    Zr = Zr._append(Ti505Ni495, ignore_index=True)
    Hf = data1.loc[data_hp['hf'] != 0].copy()
    Hf1 = Hf._append(Ti51Ni49, ignore_index=True)
    Hf49 = Hf[Hf['ni'] == 49].copy()
    Hf49.sort_values(by='hf', inplace=True, ascending=True)
    Hf = Hf._append(Ti51Ni49, ignore_index=True)
    # 搜索TiNi数据,数据存储在data2中
    data2 = pd.read_csv(filepath_or_buffer=path_hp, usecols=[i for i in range(1, 13)])
    for i in range(len(data_hp)):
        num0 = data_hp.iloc[i].value_counts().loc[0.0]  # 统计每一行0的个数
        if num0 != 9:
            data2.drop(i, inplace=True)
    data3 = pd.concat([data1, data2, data_hp])
    data3.drop_duplicates(keep=False, inplace=True)
    # 二元线性回归，得到TiNi的斜率
    x_tini = data2.loc[:, ('ti', 'ni')]
    y_tini = data2.loc[:, 'hp']
    reg_tini = LinearRegression().fit(x_tini, y_tini)
    # 将每一次更新的slope存入Slope
    Slope = pd.DataFrame()
    # 目前更新6次的slope
    # 第1代斜率，TiNi斜率设置为0
    # #调用斜率函数，求不同参杂元素的斜率
    slope1= []
    linearmodel(Cu, 'cu', slope1)
    linearmodel(Fe, 'fe', slope1)
    linearmodel(Pd, 'pd', slope1)
    linearmodel(Co, 'co', slope1)
    linearmodel(Mn, 'mn', slope1)
    linearmodel(Cr, 'cr', slope1)
    linearmodel(Nb, 'nb', slope1)
    linearmodel(Zr, 'zr', slope1)
    linearmodel(Hf, 'hf', slope1)
    slope1.insert(0, 0)  # Ni的斜率设置为0
    slope1.insert(0, 0)  # Ti的斜率设置为0
    slope1.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope1],ignore_index=True) #将slope存入Slope
    # 第2代斜率，TiNi二元体系双变量线性回归，得到Ti、Ni的斜率
    slope2 = copy.deepcopy(slope1)
    slope2[0] = reg_tini.coef_[0].tolist()
    slope2[1] = reg_tini.coef_[1].tolist()
    Slope = Slope._append([slope2],ignore_index=True)  # 将slope存入Slope
    # 第3代斜率，TiNi二元体系双变量线性回归，得到Ti、Ni的斜率，再对TiNiX三元体系减去TiNi的影响
    slope3 = []
    slope3.append(reg_tini.coef_[0].tolist())
    slope3.append(reg_tini.coef_[1].tolist())
    newlinearmodel(Cu, 'cu', slope3)
    newlinearmodel(Fe, 'fe', slope3)
    newlinearmodel(Pd, 'pd', slope3)
    newlinearmodel(Co, 'co', slope3)
    newlinearmodel(Mn, 'mn', slope3)
    newlinearmodel(Cr, 'cr', slope3)
    newlinearmodel(Nb, 'nb', slope3)
    newlinearmodel(Zr, 'zr', slope3)
    newlinearmodel(Hf, 'hf', slope3)
    slope3.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope3],ignore_index=True)  # 将slope存入Slope
    # 第4代斜率，根据ren综述中的TiNi二元体系—相变温度关系图，得到Ni的斜率，Ti设置为0
    slope4 = copy.deepcopy(slope1)
    slope4[1] = -207.49
    Slope = Slope._append([slope4],ignore_index=True)  # 将slope存入Slope
    # 第5代斜率，根据数据表中的TiNi二元体系—相变温度，得到Ni的斜率，Ti设置为0
    slope5 = copy.deepcopy(slope1)
    slope5[1] = -109.27
    Slope = Slope._append([slope5],ignore_index=True)  # 将slope存入Slope
    # 第6代斜率，在第4代基础上，将Nb分为3种替代，4元以上体系按Nb2（Nb替代Ti）计算
    # Nb的三元替代仍存在问题，Nb全部按Nb3计算？？
    slope6 = copy.deepcopy(slope4)
    slope6[8] = NbSlope(Nb)
    Slope = Slope._append([slope6], ignore_index=True)
    ##第7代斜率，TiNiX减去ren综述中的Ni斜率信息，再二元回归得到斜率
    slope7 = []
    slope7.append(0)
    slope7.append(-207.49)
    newlinearmodel2(Cu, 'cu', slope7)
    newlinearmodel2(Fe, 'fe', slope7)
    newlinearmodel2(Pd, 'pd', slope7)
    newlinearmodel2(Co, 'co', slope7)
    newlinearmodel2(Mn, 'mn', slope7)
    newlinearmodel2(Cr, 'cr', slope7)
    newlinearmodel2(Nb, 'nb', slope7)
    newlinearmodel2(Zr, 'zr', slope7)
    newlinearmodel2(Hf, 'hf', slope7)
    slope7.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope7], ignore_index=True)  # 将slope存入Slope
    ##第8代斜率，在第7代的基础上将Nb分为1、2、3
    slope8 = copy.deepcopy(slope7)
    slope8[8] = NbSlope(Nb)
    Slope = Slope._append([slope8], ignore_index=True)
    ##第9代斜率，采用Ti和Ni的差值模型，其余元素分别替代Ti和Ni
    slope9 = []
    slope9.append(reg_tini.coef_[0].tolist())
    slope9.append(reg_tini.coef_[1].tolist())
    RatioModel(Cu, 'cu', slope9)
    RatioModel(Fe, 'fe', slope9)
    RatioModel(Pd, 'pd', slope9)
    RatioModel(Co, 'co', slope9)
    RatioModel(Mn, 'mn', slope9)
    RatioModel(Cr, 'cr', slope9)
    RatioModel(Nb, 'nb', slope9)
    RatioModel(Zr, 'zr', slope9)
    RatioModel(Hf, 'hf', slope9)
    slope9.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope9], ignore_index=True)  # 将slope存入Slope
    ##1、3、7代斜率表现较好，推测下一步改变为在1代基础上修正TiNi的影响
    ##第10代斜率，TiNiX减去TiNi单变量拟合中的Ni斜率信息，再二元回归得到斜率
    slope10 = []
    slope10.append(0)
    slope10.append(-109.27)
    newlinearmodel3(Cu, 'cu', slope10)
    newlinearmodel3(Fe, 'fe', slope10)
    newlinearmodel3(Pd, 'pd', slope10)
    newlinearmodel3(Co, 'co', slope10)
    newlinearmodel3(Mn, 'mn', slope10)
    newlinearmodel3(Cr, 'cr', slope10)
    newlinearmodel3(Nb, 'nb', slope10)
    newlinearmodel3(Zr, 'zr', slope10)
    newlinearmodel3(Hf, 'hf', slope10)
    slope10.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope10], ignore_index=True)  # 将slope存入Slope
    ##第11代斜率，TiNi二元体系双变量线性回归，得到Ti、Ni的斜率，再对TiNiX三元体系减去TiNi的影响,
    # 再将Nb分为3种，即在第3代基础上改进Nb的特征
    slope11 = copy.deepcopy(slope3)
    slope11[8] = NbSlope(Nb)
    Slope = Slope._append([slope11], ignore_index=True)
    # 第12代，在11代的基础上，将Nb改进为修正斜率表征
    slope12 = []
    slope12.append(reg_tini.coef_[0].tolist())
    slope12.append(reg_tini.coef_[1].tolist())
    newlinearmodel(Cu, 'cu', slope12)
    newlinearmodel(Fe, 'fe', slope12)
    newlinearmodel(Pd, 'pd', slope12)
    newlinearmodel(Co, 'co', slope12)
    newlinearmodel(Mn, 'mn', slope12)
    newlinearmodel(Cr, 'cr', slope12)
    RatioModel(Nb, 'nb', slope12)
    newlinearmodel(Zr, 'zr', slope12)
    newlinearmodel(Hf, 'hf', slope12)
    slope12.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope12], ignore_index=True)
    # 第13代，50差值模型
    # # 为不影响原有模型，将data_hp复制后重新建模slope
    # data_hp1 = copy.deepcopy(data_hp)
    slope13 = []
    slope13.append(0)
    slope13.append(-207.49)
    newlinearmodel4(Cu, 'cu', slope13)
    newlinearmodel4(Fe, 'fe', slope13)
    newlinearmodel4(Pd, 'pd', slope13)
    newlinearmodel4(Co, 'co', slope13)
    newlinearmodel4(Mn, 'mn', slope13)
    newlinearmodel4(Cr, 'cr', slope13)
    newlinearmodel4(Nb, 'nb', slope13)
    #slope13.append(0)
    newlinearmodel4(Zr, 'zr', slope13)
    newlinearmodel4(Hf, 'hf', slope13)
    slope13.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope13], ignore_index=True)  # 将slope存入Slope
    # 第14代，50差值，Ni的斜率分段，斜率来Ms信息
    slope14 = []
    slope14.append(1.72)
    slope14.append(-207.49)
    newlinearmodel5(Cu, 'cu', slope14)
    newlinearmodel5(Fe, 'fe', slope14)
    newlinearmodel5(Pd, 'pd', slope14)
    newlinearmodel5(Co, 'co', slope14)
    newlinearmodel5(Mn, 'mn', slope14)
    newlinearmodel5(Cr, 'cr', slope14)
    newlinearmodel5(Nb, 'nb', slope14)
    #slope14.append(0)
    newlinearmodel5(Zr, 'zr', slope14)
    newlinearmodel5(Hf, 'hf', slope14)
    slope14.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope14], ignore_index=True)
    # 第15代，50差值，Ni的斜率分段，斜率来自Ap信息
    slope15 = []
    slope15.append(18.42)
    slope15.append(-109.27)
    newlinearmodel6(Cu, 'cu', slope15)
    newlinearmodel6(Fe, 'fe', slope15)
    newlinearmodel6(Pd, 'pd', slope15)
    newlinearmodel6(Co, 'co', slope15)
    newlinearmodel6(Mn, 'mn', slope15)
    newlinearmodel6(Cr, 'cr', slope15)
    newlinearmodel6(Nb, 'nb', slope15)
    #slope14.append(0)
    newlinearmodel6(Zr, 'zr', slope15)
    newlinearmodel6(Hf, 'hf', slope15)
    slope15.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope15], ignore_index=True)
    #第16代，50差值，Ni的斜率分段，斜率来自Ap信息，采用暴力搜索方法得到元素斜率
    slope16 = []
    slope16.append(18.42)
    slope16.append(-109.27)
    newlinearmodel7(Cu, 'cu', slope16)
    newlinearmodel7(Fe, 'fe', slope16)
    newlinearmodel7(Pd, 'pd', slope16)
    newlinearmodel7(Co, 'co', slope16)
    newlinearmodel7(Mn, 'mn', slope16)
    newlinearmodel7(Cr, 'cr', slope16)
    newlinearmodel7(Nb, 'nb', slope16)
    #slope14.append(0)
    newlinearmodel7(Zr, 'zr', slope16)
    newlinearmodel7(Hf, 'hf', slope16)
    slope16.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope16], ignore_index=True)
    # 第17代，50差值，Ni的斜率分段，斜率来自Ap信息，其余元素的斜率只来源于paper的fig1图
    slope17 = []
    slope17.append(18.42)
    slope17.append(-109.27)
    newlinearmodel6(Cu, 'cu', slope17)
    newlinearmodel6(Fe, 'fe', slope17)
    newlinearmodel6(Pd50, 'pd', slope17)
    newlinearmodel6(Co, 'co', slope17)
    newlinearmodel6(Mn, 'mn', slope17)
    newlinearmodel6(Cr, 'cr', slope17)
    newlinearmodel6(Nb50, 'nb', slope17)
    newlinearmodel6(Zr495, 'zr', slope17)
    newlinearmodel6(Hf49, 'hf', slope17)
    slope17.insert(11, 0)  # 功能特性的斜率设置为0
    Slope = Slope._append([slope17], ignore_index=True)
    ##汇总斜率表格，并存储
    Slope.drop(columns=11,axis=1,inplace=True)
    Slope.columns = order
    ##存储每一次更新的斜率信息
    Slope.to_pickle('Slope')

