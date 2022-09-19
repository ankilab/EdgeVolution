import tensorflow as tf
import sys
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import json
import os
import tensorflow_datasets as tfds

from utils.convert_to_tflite import convert_to_tflite
from kapre import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from utils import norm_layer

#########################################################################################
# Some general configuration
#########################################################################################
# limit GPU memory consumption to enable parallel training of multiple neural networks
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# set weights data type
#policy = tf.keras.mixed_precision.Policy('bfloat16')
#tf.keras.mixed_precision.set_global_policy(policy)

# resolve args
results_dir = sys.argv[1]
gen_dir = sys.argv[2]
nb_epochs = int(sys.argv[3])
individual_dir = sys.argv[4]


#########################################################################################
# Load data
#########################################################################################
def prepare_dataset(ds):
    # take only specific classes (0 = 'down', 1 = 'go')
    ds = ds.filter(lambda img, label: label == 2 or label == 6)
    ds = ds.map(lambda x, y: (tf.pad(x, [(0, 16000 - len(x))]), [1, 0] if y == 2 else [0, 1]))
    return ds


# TODO: specify dataset here
dataset = tfds.load("speech_commands", data_dir='datasets/', split='train', as_supervised=True, download=True)
ds_train = prepare_dataset(dataset)

dataset = tfds.load("speech_commands", data_dir='datasets/', split='test', as_supervised=True, download=True)
ds_test = prepare_dataset(dataset)


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
os.remove(model_path)  # delete the untrained TF model

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics='accuracy')

# callback for saving the best model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_trained.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


def exp_scheduler(epoch, lr):
    if epoch < 3:
        return 0.001
    else:
        return lr * np.exp(-0.1)


lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule=exp_scheduler, verbose=1)
callbacks = [lr_callback] #, model_checkpoint_callback]

# train
# TODO remove model.save because it is being saved in model checkpoint
model.fit(ds_train.batch(64), callbacks=callbacks, epochs=nb_epochs, verbose=0)

model.save(results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_trained.h5")

#########################################################################################
# Convert TF Model to TFLite Model
#########################################################################################
#model = load_tf_model(results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_trained.h5")

# TODO: use 200 mel spectrograms as representative dataset
tflite_model = convert_to_tflite(model) #, representative_data=ds_train.batch(200))

# save TFLite model
path_tflite_model = results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_tflite.tflite"
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
#test_acc = np.random.uniform(0.0, 1.0, 1)  # TODO --> placeholder
loss, test_acc = model.evaluate(ds_test.batch(64))

#########################################################################################
# Save determined test accuracy in results.json
#########################################################################################
with open(results_dir + "/" + gen_dir + "/" + individual_dir + '/results.json') as f:
    d = json.loads(f.read())

d["test_acc"] = float(test_acc)
with open(results_dir + "/" + gen_dir + "/" + individual_dir + '/results.json', 'w') as f:
    json.dump(d, f, indent=2)
