#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import pickle
import os
import json
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)
dtc_model = pickle.load(open('calModel.pkl', 'rb'))
vectorized = pickle.load(open('vectorizationCount.pkl', 'rb'))


@app.route('/spam', methods=['POST'])
def predict_row():
    content = request.json
    newData = [content['text']]
    df = pd.DataFrame(newData, columns=['text'])
    featuresTfidf = vectorized.fit_transform(df['text'])
    new_df= pd.DataFrame(featuresTfidf)
    output = dtc_model.predict(new_df)
    values = ','.join(str(v) for v in output)
    return values


if __name__ == '__main__':
	app.run(host="0.0.0.0", port=9005)