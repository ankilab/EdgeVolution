import tensorflow as tf
import sys
import numpy as np
import json
import flammkuchen as fl
import argparse

sys.path.insert(0, '.')
sys.path.insert(0, '../.')
sys.path.insert(0, '../../.')

from neural_architecture_search.src.layer_definitions import load_model
from datasets.load_data import load_dataset


def train_model(args):
    """
    Train a model with the provided arguments and save the best model in the results directory.
    :param args: Arguments for training the model
    :return: Best validation accuracy
    """
    
    # Set memory growth for GPUs to allow for dynamic memory allocation and thus training multiple models in parallel
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    #########################################################################################
    # Load data
    #########################################################################################
    ds_train, ds_val, _, class_weights = load_dataset(dataset_name=args.dataset)

    #########################################################################################
    # DNN training
    #########################################################################################
    model_path = args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + "/models/model_untrained.h5"
    model = load_model(model_path)

    model.compile(optimizer=args.optimizer,
                  loss= args.loss,
                  metrics=args.metrics)

    # callback for saving the best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + "/models/model_trained.h5",
        monitor='val_accuracy',
        mode='max',
        save_best_only=True, save_weights_only=True)

    initial_learning_rate = 0.001 
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=5,
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
    print("Training model...", flush=True)
    history = model.fit(ds_train.batch(args.batch_size),
                        validation_data=ds_val.batch(args.batch_size),
                        callbacks=callbacks,
                        verbose=0,
                        epochs=args.num_epochs,
                        class_weight=class_weights)
    print("Training finished!")
    best_val_acc = np.max(history.history['val_accuracy'])

    #########################################################################################
    # Save training history and return best validation accuracy
    #########################################################################################
    save_path = args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + "/history.fl"
    try:
        fl.save(save_path, history.history)
    except Exception:
        pass  # History save can fail if training was interrupted

    return best_val_acc

if __name__ == "__main__":
    import traceback

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
    results_path = args.results_dir + "/" + args.gen_dir + "/" + args.individual_dir + '/results.json'

    try:
        # Call the train_model function with the provided arguments
        val_acc = train_model(args)

        #####################################################################
        # Save determined val accuracy in results.json
        #####################################################################
        with open(results_path) as f:
            d = json.loads(f.read())

        d["val_acc"] = float(val_acc)

        with open(results_path, 'w') as f:
            json.dump(d, f, indent=2)

    except Exception as e:
        # Write error details to results.json so failures are diagnosable
        tb = traceback.format_exc()
        print(f"TRAINING FAILED for {args.individual_dir}: {type(e).__name__}: {e}",
              flush=True)
        print(tb, flush=True)

        try:
            with open(results_path) as f:
                d = json.loads(f.read())
        except Exception:
            d = {}

        d["training_error"] = {
            "exception": f"{type(e).__name__}: {e}",
            "traceback": tb,
        }
        with open(results_path, 'w') as f:
            json.dump(d, f, indent=2)

        sys.exit(1)