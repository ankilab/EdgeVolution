import tensorflow as tf
from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank



def get_effnet_baseline_model(input_length):
    inputs = tf.keras.Input(shape=(input_length, 1))

    efficientnet = tf.keras.applications.resnet.ResNet50(include_top=False, input_tensor=inputs)

    x = tf.keras.layers.GlobalAveragePooling2D()(efficientnet.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(13, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# def get_baseline_model(input_length):
#     inputs = tf.keras.Input(shape=(input_length, 1))

#     n_fft = 112
#     x = STFT(n_fft=n_fft, hop_length=256, input_data_format='channels_last', output_data_format='channels_last')(inputs)
#     x = Magnitude()(x)
#     x = MagnitudeToDecibel()(x)

#     kwargs = {
#         'sample_rate': 6000,
#         'n_freq': n_fft // 2 + 1,
#         'n_mels': 44,
#         'f_min': 0,
#         'f_max': 3000,
#         'htk': False,
#         'norm': 'slaney',
#     }

#     x = ApplyFilterbank(type='mel', filterbank_kwargs=kwargs, data_format='channels_last')(x)

#     x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
#     x = tf.keras.layers.MaxPooling2D()(x)
#     x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
#     x = tf.keras.layers.MaxPooling2D()(x)
#     x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
#     x = tf.keras.layers.Dense(128, activation='relu')(x)
#     outputs = tf.keras.layers.Dense(12, activation='softmax')(x)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)

#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     return model
