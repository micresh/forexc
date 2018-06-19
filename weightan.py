# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:48:39 2018

@author: Михаил Решетняк
"""
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, ActivityRegularization, BatchNormalization
from keras import optimizers

def create_x(data, window):
    res = np.array([])
    for i in range(len(data) - window):
        res = np.append(res, data[i:i+window])
    res.shape = (len(data)-window,window)
    return res

a = pd.read_csv("EURUSD1440.csv")
col_name_x ='min'
col_name_y ='max'
window_size = 7
train_size = int(len(a[col_name_x])*0.7)
train_x = create_x(a[col_name_x][0:train_size-1],window_size)
#train_x = train_x.as_matrix()
#train_x.to_csv('trainx.csv')
#train_x = create_x(train_x, window_size)
train_y = a[col_name_y][window_size:train_size-1]
#train_y.to_csv('trainy.csv')
test_x = create_x(a[col_name_x][train_size:len(a[col_name_x])], window_size)
#test_x = create_x(test_x, window_size)
test_y = a[col_name_y][train_size+window_size:len(a[col_name_y])]



#print(a['weighted'].mean())
#plt.subplot(2,1,1)
#plt.plot(a['weighted'])
#plt.subplot(2,1,2)
#plt.plot(a['max'])
#plt.show()
#data = np.random.random((2036, 12))
#print (data)
#print (len(train_x))
model = Sequential()
#model.add(LSTM(window_size, activation='sigmoid', recurrent_activation='hard_sigmoid', input_shape=(window_size,1)))
model.add(Dense(int(window_size),input_dim=window_size))
#model.add(BatchNormalization())
#model.add(ActivityRegularization(l1=0.01,l2=0.01))
#model.add(Dropout(0.3))
model.add(Dense(int(window_size*.5)))
#model.add(Dense(int(window_size*.25)))
#model.add(Dropout(0.5))
#model.add(Dense(int(window_size*.5)))
model.add(Dense(1, activation='linear'))

sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

def dev_pred(y_true, y_pred):
    return np.abs(y_pred-y_true)*2

model.compile(loss=dev_pred,
              optimizer='adam',
              metrics=[dev_pred])
history = model.fit(train_x,train_y,epochs=300)
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.show()
loss_and_metrics = model.evaluate(test_x,test_y, steps=200)
for line in loss_and_metrics:
    print(line)
pred_data = np.array([a[col_name_x][-window_size:].as_matrix()])
pred_data.shape = (1,window_size)
res = model.predict(pred_data)
print(res)
