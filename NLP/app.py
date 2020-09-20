#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import preprocessing as pp
import pandas as pd
from flask import Flask, request
import fasttext

app = Flask(__name__)
lr_model = pickle.load(open('LogisticReg_new.pkl', 'rb'))
vectorized = pickle.load(open('countvecnps.pkl', 'rb'))
fast_model=fasttext.load_model("model_filename.bin")

@app.route('/reg/sentiment', methods=['POST'])
def predict_row():
    content = request.json
    newData = [content['text']]
    df = pd.DataFrame(newData, columns=['text'])
    df = pp.preprocessingText(df)
    countvec = vectorized.transform(df['text'])
    output = lr_model.predict(countvec)
    values = ','.join(str(v) for v in output)
    return values

# @app.route('/ft/sentiment', methods=['POST'])
# def predict_row():
#     content = request.json
#     newData = [content['text']]
#     df = pd.DataFrame(newData, columns=['text'])
#     df = pp.preprocessingText(df)
#     output = fast_model.predict(df['text]'])
#     values = ','.join(str(v) for v in output)
#     return values


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9010)
