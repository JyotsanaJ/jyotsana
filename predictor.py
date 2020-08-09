#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:43:40 2020

@author: tanvi.kini
"""

import pandas as pd
import numpy as np
import seaborn as sn
import scipy.optimize as opt 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve 
from matplotlib import pyplot 
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import os
import json
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)
# model = pickle.load(open('xgb_model.pkl', 'rb'))
xg_model = pickle.load(open('xgb_model_new.pkl', 'rb'))
logistic_changed = pickle.load(open('logistic_regression.pkl', 'rb'))
minmax = pickle.load(open('minMax.pkl', 'rb'))
rf_model = pickle.load(open('random_forest.pkl', 'rb'))


@app.route('/predictnew', methods=['POST'])
def predict_row():
    content = request.json
    newData = [[content['freeMemory'], content['CpuTime'], content['ProcessCpuLoad'], content['CurrentThreadCpuTime'], content['CurrentThreaduserTime'], content['SystemLoadAverage'], content['TimeDiff']]]
    df = pd.DataFrame(newData, columns=['freeMemory', 'CpuTime', 'ProcessCpuLoad', 'CurrentThreadCpuTime','CurrentThreaduserTime', 'SystemLoadAverage', 'TimeDiff'])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    num_cols = ['freeMemory', 'CpuTime', 'ProcessCpuLoad', 'CurrentThreadCpuTime', 'CurrentThreaduserTime','SystemLoadAverage', 'TimeDiff']
    df[num_cols] = df[num_cols].values
    # X_scaled = minmax.fit(df)
    new_df = minmax.transform(df)
    new_df= pd.DataFrame(new_df)
    output = xg_model.predict(new_df)
    output2=logistic_changed.predict(new_df)
    output3=rf_model.predict(new_df)
    values = ','.join(str(v) for v in output3)
    return values


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=9002)