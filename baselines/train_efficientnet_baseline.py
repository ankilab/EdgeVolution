import tensorflow as tf
from kapre import STFT, Magnitude, MagnitudeToDecibel, ApplyFilterbank
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))  # Move up one level

# Add the parent directory to sys.path and change the current working directory
sys.path.append(parent_dir)
os.chdir(parent_dir)
from datasets.load_data import load_dataset

os.chdir(script_dir)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train_efficientnet_baseline(dataset: str):
    # check if dataset folder exists
    if not os.path.exists(dataset + "/efficientnet_baseline"):
        os.makedirs(dataset + "/efficientnet_baseline")
    
    # load dataset
    ds_train, ds_val, ds_test, class_weights = load_dataset(dataset)

    # get model and learning rate scheduler
    if dataset == 'speech_commands':
        nb_epochs = 10
        model, lr_scheduler = _get_model_speech_commands()
    elif dataset == 'emg_airob':
        nb_epochs = 30
        model, lr_scheduler = _get_model_emg_airob()
    elif dataset == 'cifar10':
        nb_epochs = 200
        model, lr_scheduler = _get_model_cifar10()
    elif dataset == 'daliac':
        nb_epochs = 60
        model, lr_scheduler = _get_model_daliac()
    else:
        raise ValueError('Invalid dataset name')
    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=lr_scheduler, verbose=0)
    
    # model checkpoint callback
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{dataset}/efficientnet_baseline/efficientnet_baseline.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True, save_weights_only=True
    )
    
    # train model
    history = model.fit(ds_train.batch(128), validation_data=ds_val.batch(128), epochs=nb_epochs, class_weight=class_weights, callbacks=[lr_callback, model_checkpoint_callback])
    
    # load best model
    model.load_weights(f'{dataset}/efficientnet_baseline/efficientnet_baseline.h5')
    
    # evaluate model (val and test set) and save results
    val_results = model.evaluate(ds_val.batch(128))
    test_results = model.evaluate(ds_test.batch(128))
    
    with open(f'{dataset}/efficientnet_baseline/efficientnet_baseline_results.txt', 'w') as f:
        f.write(f'Validation results: {val_results}\n')
        f.write(f'Test results: {test_results}\n')
        
    # save train history
    np.save(f'{dataset}/efficientnet_baseline/efficientnet_baseline_history.npy', history.history)
    
    # plot train history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f'{dataset}/efficientnet_baseline/efficientnet_baseline_accuracy_history.png')
    
    plt.clf()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f'{dataset}/efficientnet_baseline/efficientnet_baseline_loss_history.png')
    
def _get_model_cifar10():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
    efficientnet = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights=None, 
        input_shape=(32, 32, 3),
        pooling='avg'
    )

    model.add(efficientnet)
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=50,
        decay_rate=0.2,
        staircase=True)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, lr_schedule
    
def _get_model_emg_airob():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(20, 16, 1)))
    model.add(InterpolationLayer((64, 64)))

    model.add(tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3])))

    efficientnet = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=(64, 64, 3),
        pooling='avg'
    )

    model.add(efficientnet)
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(7, activation='softmax'))
    
    initial_learning_rate = 0.01 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10,
        decay_rate=0.5,
        staircase=True)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, lr_schedule
    
def _get_model_daliac():
    model = tf.keras.models.Sequential()
    model.add(STFT(input_shape=[1024, 1], n_fft=256, hop_length=32, name='stft'))
    model.add(Magnitude())
    model.add(MagnitudeToDecibel())
    kwargs = {
        'sample_rate': 200,
        'n_freq': 256 // 2 + 1,
        'n_mels': 40,
        'f_min': 0,
        'f_max': 100,
        'htk': False,
        'norm': 'slaney',
    }
    model.add(ApplyFilterbank(type='mel', filterbank_kwargs=kwargs, data_format='channels_last'))
    
    model.add(InterpolationLayer((64, 64)))
    
    # add layer that duplicates the input to 3 channels
    model.add(tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3])))

    efficientnet = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=model.layers[-1].output_shape[1:],
        pooling='avg'
    )

    model.add(efficientnet)
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(6, activation='softmax'))
    
    initial_learning_rate = 0.01 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=20,
        decay_rate=0.5,
        staircase=True)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, lr_schedule


def _get_model_speech_commands():
    model = tf.keras.models.Sequential()
    model.add(STFT(input_shape=[6000, 1], n_fft=512, hop_length=32, name='stft'))
    model.add(Magnitude())
    model.add(MagnitudeToDecibel())
    kwargs = {
        'sample_rate': 6000,
        'n_freq': 512 // 2 + 1,
        'n_mels': 128,
        'f_min': 0,
        'f_max': 3000,
        'htk': False,
        'norm': 'slaney',
    }
    model.add(ApplyFilterbank(type='mel', filterbank_kwargs=kwargs, data_format='channels_last'))

    # add layer that duplicates the input to 3 channels
    model.add(tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, 3])))

    efficientnet = tf.keras.applications.EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_shape=model.layers[-1].output_shape[1:],
        pooling='avg'
    )

    model.add(efficientnet)
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(12, activation='softmax'))
    
    initial_learning_rate = 0.001 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=0.2,
        decay_rate=0.8,
        staircase=True)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model, lr_schedule

class InterpolationLayer(tf.keras.layers.Layer):
    def __init__(self, target_size, method='bilinear', **kwargs):
        super(InterpolationLayer, self).__init__(**kwargs)
        self.target_size = target_size
        self.method = method

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method=self.method)
    

if __name__ == '__main__':
    # argparse to add dataset name
    parser = argparse.ArgumentParser(description='Train EfficientNet Baseline')
    parser.add_argument('--dataset', type=str, default='speech_commands', help='dataset name', choices=['speech_commands', 'emg_airob', 'cifar10', 'daliac', 'all'])
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = ['speech_commands', 'emg_airob', 'cifar10', 'daliac']
        for dataset in datasets:
            print(f'Training baseline efficientnet for {dataset}...')
            train_efficientnet_baseline(dataset)
    else: 
        train_efficientnet_baseline(args.dataset)