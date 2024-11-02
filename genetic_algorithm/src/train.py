import tensorflow as tf
import sys
import numpy as np
import json
import flammkuchen as fl
from kapre import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
import argparse

sys.path.insert(0, '.')
sys.path.insert(0, '../.')
sys.path.insert(0, '../../.')

from genetic_algorithm.utils.convert_to_tflite import convert_to_tflite
from genetic_algorithm.utils.substitute_tflite_layer import substitute_tflite_layer
from genetic_algorithm.utils import norm_layer
from datasets.load_data import get_datasets

# from genepool_modules.sinc_conv_layer import SincConv1D
# from tensorflow_addons.layers import InstanceNormalization


def train_model(args):
    #########################################################################################
    # Some general configuration
    #########################################################################################
    # limit GPU memory consumption to enable parallel training of multiple neural networks
    # --> Memory limit is not really 1024MB in practice...
    #gpus = tf.config.list_physical_devices('GPU')
    #tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

    # --> This is better, but we never know how many memory exactly is allocated
    # However, 24 GB should be enough to train 10 models in parallel, even if we have 10 huge models
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    #########################################################################################
    # Load data
    #########################################################################################
    params = {'input_shape': args.input_shape,
              'classes_filter': args.classes_filter}

    ds_train, ds_val, _, class_weights = get_datasets(dataset=args.dataset, params=params)

    #########################################################################################
    # DNN training
    #########################################################################################
    # load and compile tf model
    def load_tf_model(path):
        m = tf.keras.models.load_model(path, custom_objects={'STFT': STFT,
                                                             'Magnitude': Magnitude,
                                                             'ApplyFilterbank': ApplyFilterbank,
                                                             'MagnitudeToDecibel': MagnitudeToDecibel})
        return m

    model_path = args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + "/models/model_untrained.h5"
    model = load_tf_model(model_path)

    model.compile(optimizer=args.optimizer,
                  loss= args.loss,
                  metrics=args.metrics)

    # callback for saving the best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + "/models/model_trained.h5",
        monitor='val_accuracy',
        mode='max',
        save_best_only=True, save_weights_only=True)

    def exp_scheduler(epoch, lr):
        if epoch < 10:
            return 0.001
        elif epoch < 20:
            return 0.0005
        elif epoch < 30:
            return 0.0001
        else:
            return lr * np.exp(-0.1)

    def daliac_scheduler(epoch, lr):
        if epoch < 75:
            return 0.001
        elif epoch < 125:
            return 0.0005
        elif epoch < 175:
            return 0.0001
        else:
            return lr * np.exp(-0.1)

    initial_learning_rate = 0.001 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=0.2,
        decay_rate=0.8,
        staircase=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=5,
        restore_best_weights=True)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=0)
    callbacks = [lr_callback, model_checkpoint_callback, early_stopping]

    # train
    print("Training model...")
    try:
        history = model.fit(ds_train.batch(args.batch_size),
                            validation_data=ds_val.batch(args.batch_size),
                            callbacks=callbacks,
                            # verbose=0,
                            epochs=args.num_epochs, 
                            class_weight=class_weights)
        print("Training finished!")
        best_val_acc = np.max(history.history['val_accuracy'])
    except Exception as e:
        print(f"Exception during training: {e}")
        best_val_acc = 0

    #########################################################################################
    # Save training history
    #########################################################################################
    save_path = args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + "/history.fl"
    try:
        fl.save(save_path, history.history)
    except:
        # History is not existing as something went wrong during training
        pass

    return best_val_acc


# resolve args
parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str)
parser.add_argument("--gen_dir", type=str)
parser.add_argument("--individual_dir", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--classes_filter", type=int, nargs="*")
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--input_shape", type=int, nargs="*")
parser.add_argument("--loss", type=str)
parser.add_argument("--metrics", type=str, nargs="*")
parser.add_argument("--optimizer", type=str)

args = parser.parse_args()

# Call the train_model function with the provided arguments

val_acc = train_model(args)

if val_acc == -1:
    # Training failed in this case, try it one more time now
    val_acc = train_model(args)

#########################################################################################
# Save determined val accuracy in results.json
#########################################################################################
with open(args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + '/results.json') as f:
    d = json.loads(f.read())

try:
    d["val_acc"] = float(val_acc)
except:
    d["val_acc"] = val_acc

with open(args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + '/results.json', 'w') as f:
    json.dump(d, f, indent=2)