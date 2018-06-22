# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:48:39 2018

@author: Михаил Решетняк
"""
import numpy as np

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, ActivityRegularization, BatchNormalization
from keras import optimizers


def create_x(data, window):
    res = np.array([])
    for i in range(len(data) - window):
        res = np.append(res, data[i:i+window])
    res.shape = (len(data)-window, window)
    return res


def ann_pred(ws, epochs):
    a = pd.read_csv("EURUSD1440-220618.csv")
    col_name_x = 'close'
    col_name_y = 'close'
    window_size = ws
    train_size = int(len(a[col_name_x]) * 0.7)
    train_x = create_x(a[col_name_x][0:train_size - 1], window_size)
    train_y = a[col_name_y][window_size:train_size - 1]
    test_x = create_x(a[col_name_x][train_size:len(a[col_name_x])], window_size)
    test_y = a[col_name_y][train_size + window_size:len(a[col_name_y])]

    model = Sequential()
    model.add(Dense(int(window_size), input_dim=window_size))
    model.add(Dense(int(window_size * 10)))
    model.add(Dense(int(window_size * 5)))
    model.add(Dense(int(window_size)))
    model.add(Dense(1, activation='linear'))

    def dev_pred(y_true, y_pred):
        return np.abs(y_pred - y_true) * 2

    model.compile(loss=dev_pred,
                  optimizer='nadam',
                  metrics=['mae'])
    history = model.fit(train_x, train_y, epochs=epochs)
    loss_and_metrics = model.evaluate(test_x, test_y, steps=200)
    pred_data = np.array([a[col_name_x][-window_size:].as_matrix()])
    pred_data.shape = (1, window_size)
    resu = model.predict(pred_data)
    return loss_and_metrics, resu


predict = ann_pred(4, 150)
print(predict[0][1], '    ', predict[1][0])