# -*- coding:utf-8
"""
author: solid
data : 20181213
#description: GBDT回归模型调参
"""
# ############
# 参数择优
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import random
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_friedman1
from sklearn.model_selection import GridSearchCV
# ##############
X, y = make_friedman1(n_samples=1200, random_state=1, noise=1.0)
# ###########
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(X, y,
                                                                             random_state=2)
# 利用默认参数，对原始数据进行回归
clf = GradientBoostingRegressor(loss='quantile')
clf.fit(diabetes_X_train,diabetes_y_train)
y_pred = clf.predict(diabetes_X_test)
# ##########
#评价模型的均方差情况
mse_Default = mean_squared_error(y_pred,diabetes_y_test)
print 'mse_Default = %f' %mse_Default
# 对boosting框架参数：对损失函数huber的分位数α和权重缩放率learning_rate进行调参
param_test1 = {'alpha':np.linspace(0.3,0.9,7),
               'learning_rate':np.linspace(0.2,0.9,8)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=100,
                                            loss='quantile',random_state=10),
                       param_grid = param_test1,iid=False,cv=5)
gsearch1.fit(diabetes_X_train,diabetes_y_train)
print gsearch1.best_params_,gsearch1.best_score_
#用最佳参数拟合训练数据得到新模型
clf = GradientBoostingRegressor(loss='quantile',alpha=0.5,learning_rate=0.2)
clf.fit(diabetes_X_train,diabetes_y_train)
#用该模型预测测试数据，得到均方差
y_pred1 = clf.predict(diabetes_X_test)
mse_ParamOpt = mean_squared_error(y_pred1,diabetes_y_test)
print 'mse_ParamOpt = %f' %mse_ParamOpt