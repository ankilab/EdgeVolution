import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank

# Change working directory to the root of the project to be able to import the necessary modules
import os
os.chdir('../.')

import sys
sys.path.append('.')
sys.path.append('../')
from datasets.load_data import load_dataset

ds_train, ds_val, ds_test, _ = load_dataset(dataset_name="cifar10")

model = tf.keras.models.Sequential()

efficientnet = tf.keras.applications.EfficientNetB0(
    include_top=False, 
    weights=None, 
    input_shape=(32, 32, 3),
    pooling='avg'
)

model.add(efficientnet)
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=75,
    decay_rate=0.75,
    staircase=True)

lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=0)


# save best model callback
save_best = tf.keras.callbacks.ModelCheckpoint('baselines/best_model_cifar10.h5', monitor='val_accuracy', save_best_only=True)

history = model.fit(ds_train.batch(64), validation_data=ds_val.batch(64), epochs=100, callbacks=[lr_callback, save_best])

# save history
np.save('baselines/history_cifar10.npy', history.history)