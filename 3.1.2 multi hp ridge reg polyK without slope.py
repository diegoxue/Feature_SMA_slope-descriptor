import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
import copy
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.kernel_ridge import KernelRidge

def RidgeResample(x,y,model):
    mse_FIT, mse_PRE, mse_fit_min, mse_pre_min = [], [], 10 ** 6, 10 ** 6
    #分割100次测试集和训练集
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
        reg = model.fit(x_train, y_train)
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

#获取CTestMeanAddStd字典中的最小的15个,
def get15(CTestMean):
    key15=[]
    CTestMean2 = CTestMean.copy()
    for i in range(15):
        #浅拷贝，修改CTestMean2不会改变CTestMean
        #获得最小值对应的键
        key = min(CTestMean2, key=CTestMean2.get)
        key15.append(key)
        #删除此最小值
        CTestMean2.pop(key)
    return key15



if __name__ == '__main__':
    slope_str,slope_num = 'slope15', 14
    data_hp=pd.read_pickle('hp_tini')
    hp_slope = pd.read_pickle('hp_slope_tini')
    slope_imp = hp_slope.loc[:,slope_str]
    Slope = pd.read_pickle('Slope')
    Feature = ['slope_imp','mr','ar_c','en','anum','ven','ea','dor','mass','volume','energy1','CE','EBE','YM','cs','Tm','cv']
    data_hp.iloc[:,1] = slope_imp
    data_hp.rename(columns = {'slope1':'slope_imp'}, inplace=True)
    x_slope = data_hp['slope_imp'].values.reshape(-1,1)
    y = data_hp['hp'].values.reshape(-1,1)
    # 删除slope_imp特征
    Feature.remove('slope_imp')
    data_hp.drop(labels='slope_imp',axis=1,inplace=True)


    # # 比较第15代斜率特征和其他特征的多元RR和LR拟合对比
    # # 组合1-6个特征进行岭回归，并提取前15特征组合
    CompTrainMean,CompTestMean,CompTrainStd,CompTestStd,Compk = [],[],[],[],[]
    for m in tqdm(range(5,7)):
        Feature_inter = combinations(Feature,m)
        CTrainMean, CTrainStd, CTestMean, CTestStd, Creg, CTestMeanAddStd = {}, {}, {}, {}, {}, {}
        PlotCTrainMean, PlotCTrainStd, PlotCTestMean, PlotCTestStd = [], [], [], []
        for i in tqdm(Feature_inter):
            # 分析hp的情况
            X = data_hp.loc[:,i]
            X_scaler = StandardScaler().fit_transform(X)
            y = data_hp.loc[:,'hp']
            x_train, x_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=10)
            params_poly = {'kernel': ['poly'], 'alpha': [i for i in np.arange(0.01, 10, 0.1)],
                           'gamma': [i for i in np.arange(0.1, 2, 0.1)], 'coef0': [0.01, 0.1, 1, 100],
                           'degree': [1, 2, 3]}
            bestscore = -10000
            for m in range(10):
                krr_poly = RandomizedSearchCV(KernelRidge(), params_poly, cv=10, verbose=0,
                                              scoring='neg_mean_squared_error', n_iter=30,
                                              random_state=m,n_jobs=10)  # cv=10即10折交叉验证
                krr_poly.fit(x_train, y_train)  # 对给定数据集选取最佳参数
                if krr_poly.best_score_>bestscore:
                    bestscore = krr_poly.best_score_
                    model = krr_poly.best_estimator_
            results = RidgeResample(X_scaler,y,model)
            #Creg[Feature[i]+'',''+Feature[j]] = results['reg']
            CTrainMean[i] = np.mean(results['mse_FIT'])
            CTestMean[i] = np.mean(results['mse_PRE'])
            CTrainStd[i] = np.std(results['mse_FIT'])
            CTestStd[i] = np.std(results['mse_PRE'])
            CTestMeanAddStd[i] = CTestMean[i]+CTestStd[i]
        #获取CTestMean字典中的最小的15个对应的键
        key15 = get15(CTestMeanAddStd)
        CompTrainMean.append(CTrainMean[key15[0]])
        CompTrainStd.append(CTrainStd[key15[0]])
        CompTestMean.append(CTestMean[key15[0]])
        CompTestStd.append(CTestStd[key15[0]])
        Compk.append(CTrainMean[key15[0]])
        # 按CTestMean字典中的最小的15个的键提取对应的CTrainMean,CTrainStd,CTestMean
        for k in key15:
            PlotCTrainMean.append(CTrainMean[k])
            PlotCTestMean.append(CTestMean[k])
            PlotCTrainStd.append(CTrainStd[k])
            PlotCTestStd.append(CTestStd[k])
        #绘图1
        fig, ax = plt.subplots(1, 1)
        # 设置图例和坐标轴加粗,字体加大
        plt.rcParams["font.weight"] = 'bold'
        plt.rcParams["axes.labelweight"] = 'bold'
        plt.rcParams["font.size"] = 10
        #柱状图中柱的数目以及柱宽
        index, bar_width = np.arange(len(PlotCTestMean)), 0.3
        ax.bar(index, PlotCTrainMean, bar_width, yerr=PlotCTrainStd, capsize=4, alpha=0.5, color='g', ecolor='g',
                  ls='-', lw=1, ec='g')
        # index即x轴的位置, mse_train_mean即y轴的位置, bar_width是柱状图的宽度,yerr是设定误差值,capsize即误差杠的长度
        # alpha透明度, color柱状图填充颜色,ecolor误差杠颜色,ls柱状图轮廓线形, lw柱状图轮廓线粗细,ec柱状图轮廓线颜色
        ax.bar(index + bar_width, PlotCTestMean, bar_width, yerr=PlotCTestStd, capsize=4, alpha=0.5, color='r',
                  ecolor='r', ls='-', lw=1, ec='r')
        ax.legend(labels=['mse_train', 'mse_test'], loc='best')
        ax.set_ylabel('MSE')
        # 修改x轴坐标要先给定坐标轴，再修改
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(key15)
        plt.xticks(rotation=45)
        plt.show()
    #绘图2，绘制1-6元特征的拟合效果比较图
    fig, ax = plt.subplots(1, 1)
    # 设置图例和坐标轴加粗,字体加大
    plt.rcParams["font.weight"] = 'bold'
    plt.rcParams["axes.labelweight"] = 'bold'
    plt.rcParams["font.size"] = 10
    # 柱状图中柱的数目以及柱宽
    index, bar_width = np.arange(len(CompTrainMean)), 0.3
    ax.bar(index, CompTrainMean, bar_width, yerr=CompTrainStd, capsize=4, alpha=0.5, color='g', ecolor='g',
           ls='-', lw=1, ec='g')
    # index即x轴的位置, mse_train_mean即y轴的位置, bar_width是柱状图的宽度,yerr是设定误差值,capsize即误差杠的长度
    # alpha透明度, color柱状图填充颜色,ecolor误差杠颜色,ls柱状图轮廓线形, lw柱状图轮廓线粗细,ec柱状图轮廓线颜色
    ax.bar(index + bar_width, CompTestMean, bar_width, yerr=CompTestStd, capsize=4, alpha=0.5, color='r',
           ecolor='r', ls='-', lw=1, ec='r')
    for a,b in zip(index,CompTrainMean):
        ax.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7)
    for a,b in zip(index+bar_width,CompTestMean):
        ax.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7)
    ax.legend(labels=['mse_train', 'mse_test'], loc='best')
    ax.set_xlabel('Model complexity')
    ax.set_ylabel('MSE')
    # 修改x轴坐标要先给定坐标轴，再修改
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([i for i in range(1,len(CompTrainMean)+1)])
    plt.show()





