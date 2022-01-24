# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:15:45 2021

@author: Yaren
"""

import numpy as np
import pandas as pd
import tensorflow
import os

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

bus_dir = os.path.join("train/Bus")
helicopter_dir = os.path.join("train/Helicopter")
motorcycle_dir = os.path.join("train/Motorcycle")
taxi_dir = os.path.join("train/Taxi")

bus_names = os.listdir(bus_dir)
helicopter_names = os.listdir(helicopter_dir)
motorcycle_names = os.listdir(motorcycle_dir)
taxi_names = os.listdir(taxi_dir)

batch_size = 64
epochs = 1
data_aguamentation = True


train_datagenerator = ImageDataGenerator(rescale = 1./255,
                                         featurewise_std_normalization=True,
                                         rotation_range = 40,
                                         width_shift_range = 0.2,
                                         height_shift_range = 0.2,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         fill_mode='nearest')

train_data = train_datagenerator.flow_from_directory("train", 
                                                    target_size=(150,150),
                                                    batch_size = batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical')


val_datagenerator = ImageDataGenerator(rescale=1./255)
validation_data = val_datagenerator.flow_from_directory("validation", 
                                                  target_size=(150,150), 
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  class_mode='categorical')

test_datagenerator = ImageDataGenerator(rescale=1./255)
test_data = test_datagenerator.flow_from_directory("test", 
                                                  target_size=(150,150), 
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  class_mode='categorical')

train_number = train_data.samples
validation_number = validation_data.samples 

model = Sequential()

model.add(Conv2D(32, (5,5), activation = "relu", input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (5,5),  activation = "relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3,3),  activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3),  activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3),  activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(4, activation = "softmax"))
model.compile(optimizer= "adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit_generator(train_data, 
                    epochs=epochs,
                    steps_per_epoch = train_number // batch_size,
                    validation_data= validation_data,
                    validation_steps = validation_number // batch_size,
                    verbose = 2)

ypred = model.predict(test_data)
print("Accuracy Score:", model.evaluate(test_data))




