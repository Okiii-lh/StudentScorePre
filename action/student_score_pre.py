#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File      :   student_score_pre.py
@Contact   :   okery.github.io

@Modify Time        @Author     @Version    @Description
------------        -------     --------    ------------
2019/10/15上午10:43  LiuHe       1.0         基于逻辑回归预测学生成绩
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
import sndhdr as sr

# 过滤掉无用的警告
warnings.filterwarnings('ignore')

df = pd.read_csv("../dataset/StudentPerformance.csv")

# 检测数据集大小
print("数据集大小")
print(df.shape)

# 查看数据的相关信息

print("前五行信息---------")
print(df.head())
print("------------------")
print("空数据数量---------")
print(df.isnull().sum())
print("------------------")
print("数据描述-------------")
print(df.describe())
print(df.info())
print("-------------------")
print("数据类型-------------")
print(df.dtypes)
print("----------------------")
print("分类特征--------------")
print(df['Class'].unique())


# 查看那数据分布是否均衡
class_height = [
    df[df['Class'] == 'M'].count()[0],
    df[df['Class'] == 'L'].count()[0],
    df[df['Class'] == 'H'].count()[0]
]
plt.bar(df['Class'].unique(), height=class_height)
plt.show()

# 查看数据男女分布是否均衡
gender_height = [
    df[df['gender'] == 'M'].count()[0],
    df[df['gender'] == 'F'].count()[0]
]
plt.bar(df['gender'].unique(), height=gender_height)
plt.show()

test_x = df.drop('Class', axis=1)
test_x = test_x.drop('PlaceofBirth', axis=1)
test_y = df['Class']
# 将所有非数值型变量转化为数值型
test_x = pd.get_dummies(test_x)

x_train, x_test, y_train, y_test \
    = train_test_split(test_x, test_y, test_size=0.2, random_state=10)

lr = LogisticRegression()
lr.fit(x_train, y_train)

predict_y = lr.predict(x_test)
print("predict", predict_y)

score = accuracy_score(y_test, predict_y)

print("score", score)



