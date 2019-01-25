# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 00:27:27 2019

@author: Suraj
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

data = np.random.random((1000,100))
labels = np.random.randint(2, size = (1000,1))

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(data, labels, epochs=10, batch_size=32)

prediction = model.predict(data)
print("\n")
print(np.average(prediction))