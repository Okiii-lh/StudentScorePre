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
import matplotlib as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
import sndhdr as sr

# 过滤掉无用的警告
warnings.filterwarnings('ignore')


