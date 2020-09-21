# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 18:42:30 2020

@author: Mingcong Li
"""

import numpy as np
import pandas as pd
import os
from sklearn import metrics


# 改变工作目录
os.chdir(r"C:\Users\Mingcong Li\Desktop\携程网点评内容情感分析")
# 读取数据
test_pred = np.load("test_pred.npy")
y_test = np.load("y_test.npy")
y_train = np.load("y_train.npy")
# 包含字符串元素的array，每一个元素都是object，需要加上allow_pickle=True。默认是False，防止读入恶意信息。
X_test = np.load("X_test.npy", allow_pickle=True)
X_train = np.load("X_train.npy", allow_pickle=True)




unique, counts = np.unique(y_train, return_counts=True)  # 检查一下分的是否均匀
dict(zip(unique, counts))



# 评估模型预测结果
# 准确率
metrics.accuracy_score(y_test, test_pred)
# 召回率
metrics.recall_score(y_test, test_pred, average='binary')
# precision
metrics.precision_score(y_test, test_pred, average='binary')
# 混淆矩阵
metrics.confusion_matrix(y_test, test_pred)
# 计算ROC值
metrics.roc_auc_score(y_test, test_pred)
