import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import numpy as np
import itertools

sys.path.append('../../datasets')
from load_data import get_datasets

from model import get_baseline_model

def grid_search(n_ffts, hop_lengths, n_mels):
    # Load the speech commands dataset
    params = {
        'input_shape': (6000, 1),
        'classes_filter': [],
        'num_classes': 12
    }

    ds_train, ds_val, ds_test, class_weights = get_datasets("speech_commands", params)

    # create file to save results
    f = open('results.txt', 'w')

    # Get the model
    for n_fft, hop_length, n_mel in itertools.product(n_ffts, hop_lengths, n_mels):
        model = get_baseline_model(params['input_shape'][0], n_fft, hop_length, n_mel)

        checkpoint_filepath = f'checkpoint_{n_fft}_{hop_length}_{n_mel}.h5'
        model_checkpoint_callback = get_checkpoint_callback(checkpoint_filepath)
        lr_scheduler = get_lr_scheduler()

        # Train the model
        history = model.fit(ds_train.batch(64), validation_data=ds_val.batch(64), 
                            epochs=10, class_weight=class_weights, 
                            callbacks=[model_checkpoint_callback, lr_scheduler])

        # Plot the training history
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.savefig(f'history_{n_fft}_{hop_length}_{n_mel}.png')

        # Evaluate the model
        test_loss, test_acc = model.evaluate(ds_test.batch(64), verbose=2)
        
        # Save the results
        f.write(f'n_fft: {n_fft}, hop_length: {hop_length}, n_mel: {n_mel}, test_loss: {test_loss}, test_acc: {test_acc}\n')

    f.close()

def get_lr_scheduler():
    return tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 * 0.5 ** epoch)

def get_checkpoint_callback(filename):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filename,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)


if __name__ == "__main__":
    n_ffts = [128, 256, 512, 1024, 2048]
    hop_lengths = [128, 256, 396]
    n_mels = [40, 64, 128]

    grid_search(n_ffts, hop_lengths, n_mels)