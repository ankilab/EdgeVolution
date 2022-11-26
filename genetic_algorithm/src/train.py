import tensorflow as tf
import sys
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import json
import os
import flammkuchen as fl
from kapre import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel

sys.path.insert(0, '.')
sys.path.insert(0, '../.')
sys.path.insert(0, '../../.')

from genetic_algorithm.utils.convert_to_tflite import convert_to_tflite
from genetic_algorithm.utils.substitute_tflite_layer import substitute_tflite_layer
from genetic_algorithm.utils import norm_layer
from datasets.get_datasets import get_datasets

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

# set weights data type
# policy = tf.keras.mixed_precision.Policy('bfloat16')
# tf.keras.mixed_precision.set_global_policy(policy)

# resolve args
results_dir = sys.argv[1]
gen_dir = sys.argv[2]
nb_epochs = int(sys.argv[3])
dataset = sys.argv[4]
individual_dir = sys.argv[5]

#########################################################################################
# Load data
#########################################################################################
ds_train, ds_val, ds_test = get_datasets(dataset)


#########################################################################################
# DNN training
#########################################################################################
# load and compile tf model
def load_tf_model(path):
    m = tf.keras.models.load_model(path, custom_objects={"InstanceNormalization": InstanceNormalization,
                                                         'NormLayer': norm_layer.NormLayer,
                                                         'STFT': STFT,
                                                         'Magnitude': Magnitude,
                                                         'ApplyFilterbank': ApplyFilterbank,
                                                         'MagnitudeToDecibel': MagnitudeToDecibel})
    return m


model_path = results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_untrained.h5"
model = load_tf_model(model_path)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# callback for saving the best model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_trained.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True, save_weights_only=True)


def exp_scheduler(epoch, lr):
    if epoch < 2:
        return 0.01
    elif epoch < 4:
        return 0.001
    elif epoch < 6:
        return 0.0001
    else:
        return lr * np.exp(-0.1)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.5,
    patience=3,
    verbose=0,
    mode="max",
    baseline=None,
    restore_best_weights=False,
)

lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=exp_scheduler, verbose=0)
callbacks = [lr_callback, model_checkpoint_callback, early_stopping]

# train
history = model.fit(ds_train.batch(128),
                    validation_data=ds_val.batch(64),
                    callbacks=callbacks,
                    verbose=0,
                    epochs=nb_epochs)

#########################################################################################
# Save training history
#########################################################################################
save_path = results_dir + "/" + gen_dir + "/" + individual_dir + "/history.fl"
fl.save(save_path, history.history)

#########################################################################################
# Convert TF Model to TFLite Model
#########################################################################################
model = load_tf_model(results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_untrained.h5")
model.load_weights(results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_trained.h5")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# TODO: use 200 mel spectrograms as representative dataset
tflite_model = substitute_tflite_layer(model)
tflite_model = convert_to_tflite(tflite_model, np.random.uniform(size=(200, 6000, 1)))

# save TFLite model
path_tflite_model = results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_tflite_trained.tflite"
with open(path_tflite_model, "wb") as fp:
    fp.write(tflite_model)

#########################################################################################
# Determine test accuracy using TF Lite model
#########################################################################################

# input_details = tflite_model.get_input_details()
# output_details = tflite_model.get_output_details()
# tflite_model.allocate_tensors()
#
# for x_test in X_test:
#     tflite_model.set_tensor(input_details[0]['index'], [x_test])
#     tflite_model.invoke()
#     output_data = tflite_model.get_tensor(output_details[0]['index'])

# TODO compare normal model output and TFLite model output

loss, test_acc = model.evaluate(ds_test.batch(64))

#########################################################################################
# Save determined test accuracy in results.json
#########################################################################################
with open(results_dir + "/" + gen_dir + "/" + individual_dir + '/results.json') as f:
    d = json.loads(f.read())

d["test_acc"] = float(test_acc)
with open(results_dir + "/" + gen_dir + "/" + individual_dir + '/results.json', 'w') as f:
    json.dump(d, f, indent=2)
