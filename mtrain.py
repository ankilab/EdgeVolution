import tensorflow as tf
import sys
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

sys.path.insert(0, '.')
sys.path.insert(0, '../.')
sys.path.insert(0, '../../.')

import tensorflow_datasets as tfds

EPOCHS = 200
ds_train, ds_test = tfds.load('cifar10', split=['train[:2%]', 'test[:1%]'], as_supervised=True)

ds_train = ds_train.batch(128)
ds_test = ds_test.batch(128)

model = tf.keras.Sequential()

# model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
#model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(MaxPooling2D((2, 2)))


# example output part of the model
model.add(Flatten())
#model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))






# model.add(tf.keras.layers.Flatten(input_shape=(32, 32, 3)))
# model.add(tf.keras.layers.Dense(100, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
opt = SGD(lr=0.001, momentum=0.9)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics='accuracy')


history = model.fit(ds_train,
                    validation_data=ds_test,
                    epochs=EPOCHS,
)

