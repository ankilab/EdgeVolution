import tensorflow as tf
from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank



def get_baseline_model(input_length):
    inputs = tf.keras.Input(shape=(input_length, 1))

    x = STFT(n_fft=1024, hop_length=128, input_data_format='channels_last', output_data_format='channels_last')(inputs)
    x = Magnitude()(x)
    x = MagnitudeToDecibel()(x)

    kwargs = {
        'sample_rate': 6000,
        'n_freq': n_fft // 2 + 1,
        'n_mels': 40,
        'f_min': 0,
        'f_max': 3000,
        'htk': False,
        'norm': 'slaney',
    }

    x = ApplyFilterbank(type='mel', filterbank_kwargs=kwargs, data_format='channels_last')(x)

    efficientnet = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=x, weights=None)

    x = tf.keras.layers.GlobalAveragePooling2D()(efficientnet.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(12, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='ategorical_crossentropy', metrics=['accuracy'])

    return model
