# -*- coding:utf-8
"""
#Author: solid
#data: 20181213
#description: 探讨回归模型损失函数对数据的灵敏度
"""
# ############
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import random
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1
# ##############
# 构造数据
X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
# ########
# 加入噪声,定义均值和方差
mu = 0
sigma = 2
X2 = np.zeros(X.shape)
X3 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X2[i,j] = X[i,j] + random.uniform(mu,sigma)
        X3[i,j] = X[i,j]
# #############
# 对因变量添加异常值
y1 = np.zeros(y.shape)
start = 90
stop = 100
for i in range(y.shape[0]):
    if(i%60 ==0):
        y1[i] = y[i] + random.uniform(start,stop)   # 对因变量增加[90,100]的均匀分布
# ######
# 输入变量和输出变量写成list形式
X1 =[X,X2,X3]
y0 =[y,y,y1]
# ###########
# 根据不同的损失函数来拟合
params1 = {'n_estimators':500,'loss':'ls','learning_rate':0.8}
params2 = {'n_estimators':500,'loss':'lad','learning_rate':0.8}    #lad
params3 = {'n_estimators':500,'loss':'huber','learning_rate':0.8}  # huber
params4 = {'n_estimators':500,'loss':'quantile','learning_rate':0.8} #quantile,0.2,0.5
clf1 = GradientBoostingRegressor(**params1)         # 均方差
clf2 = GradientBoostingRegressor(**params2)        # 绝对损失
clf3 = GradientBoostingRegressor(**params3)      # huber损失
clf4 = GradientBoostingRegressor(**params4)   # quantile损失
# ########
# 求训练模型对不同数据类型的敏感度
dataType =['org_data','noisy_data','outlier_data']
color1='grb'
plt.figure()
for k,clf,name,c1 in [(1,clf1,'ls','g'),(2,clf2,'lad','r'),(3,clf3,'huber','b'),(4,clf4,'quanitle','k')]:
    plt.subplot(2,2,k)
    for org_data,y5,name1,c2 in zip(X1,y0,dataType,color1):
    # 分成训练数据和测试数据4:1比例
        diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(org_data, y5,
                                                                                            random_state=2)
        clf.fit(diabetes_X_train, diabetes_y_train)
        test_score = np.zeros((params1['n_estimators'],), dtype=np.float64)
        # 输出每次迭代的均方差结果
        for i, y_pred in enumerate(clf.staged_predict(diabetes_X_test)):
            test_score[i] = mean_squared_error(diabetes_y_test, y_pred)
        plt.plot(np.arange(params1['n_estimators']) + 1, test_score, c=c2,
                     label=name1)
        plt.subplots_adjust(hspace = 0.5)
        plt.subplots_adjust(wspace = 0.5)
        plt.legend()
        plt.xlabel('Boosting Iterations')
        plt.ylabel('Deviance')
        plt.title(name)
plt.show()