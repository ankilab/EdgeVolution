"""
Script to download EfficientNet model from TensorFlow. 
"""

import tensorflow as tf

efficientnet = tf.keras.applications.EfficientNetB0(
    include_top=False, 
    weights='imagenet', 
    input_shape=(224, 224, 3),
    pooling='avg'
)

print(efficientnet.summary())