import tensorflow as tf
from tensorflow.keras import mixed_precision
import sys
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import json

from utils.convert_to_tflite import convert_to_tflite

#########################################################################################
# Some general configuration
#########################################################################################
# limit GPU memory consumption to enable parallel training of multiple neural networks
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# set weights data type
policy = mixed_precision.Policy('bfloat16')
mixed_precision.set_global_policy(policy)

# resolve args
results_dir = sys.argv[1]
gen_dir = sys.argv[2]
individual_dir = sys.argv[3]

#########################################################################################
# Load data
#########################################################################################
# TODO
X_train = None
Y_train = None
X_test = None
Y_test = None


#########################################################################################
# DNN training
#########################################################################################
# load and compile tf model
def load_tf_model(path):
    m = tf.keras.models.load_model(path, custom_objects={"InstanceNormalization": InstanceNormalization})
    return m


model = load_tf_model(results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_untrained.h5")
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='CategoricalCrossentropy')

# callback for saving the best model
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_trained.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# train
# TODO remove model.save because it is being saved in model checkpoint
# model.fit(X_train, Y_train, callbacks=[model_checkpoint_callback])
model.save(results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_trained.h5")

#########################################################################################
# Convert TF Model to TFLite Model
#########################################################################################
model = load_tf_model(results_dir + "/" + gen_dir + "/" + individual_dir + "/models/model_trained.h5")

# TODO: use 200 mel spectrograms as representative dataset
tflite_model = convert_to_tflite(model, representative_data=None)

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
test_acc = np.random.uniform(0.0, 1.0, 1)  # TODO --> placeholder

#########################################################################################
# Save determined test accuracy in results.json
#########################################################################################
with open(results_dir + "/" + gen_dir + "/" + individual_dir + '/results.json') as f:
    d = json.loads(f.read())

d["test_acc"] = float(test_acc)
with open(results_dir + "/" + gen_dir + "/" + individual_dir + '/results.json', 'w') as f:
    json.dump(d, f, indent=2)
