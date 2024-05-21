import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../datasets')
from load_data import get_datasets

from model import get_effnet_baseline_model


def train():
    ds_train, ds_val, ds_test, class_weights = get_datasets("daliac", "../../datasets/daliac/", return_one_hot=True)

    #model = get_effnet_baseline_model((2024, 24, 1))

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(2048, 24, 1), padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(13, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    model.fit(ds_train.batch(64), validation_data=ds_val.batch(64), epochs=5, class_weight=class_weights)

    model.save("/home/woody/iwb3/iwb3022h/EvoNAS_Results/daliac_model.h5")

if __name__ == '__main__':
    train()