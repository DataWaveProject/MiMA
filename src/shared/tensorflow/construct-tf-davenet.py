import argparse
from pathlib import Path

import numpy as np
import torch

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense


def main():
    '''
    Construct a TF model that looks like davenet.
    '''

    ap = argparse.ArgumentParser()
    ap.add_argument('davenet_file', type=argparse.FileType('rb'))
    ap.add_argument(
        '--output_dir',
        '-o',
        type=Path,
        default=Path('saved_model')
    )
    args = ap.parse_args()

    # First load the PyTorch davenet weights, etc
    # checkpoint is a dict, top-level keys are 'weights', 'means', 'stds'.
    # Latter two are for final normalisation.

    checkpoint = torch.load(args.davenet_file,
                            map_location=torch.device('cpu'))


    the_type = tf.float64

    # Newest davenet (as modified by Jack Atkinson) has 3 Inputs,
    # wind, lat, pressure.
    # wind is (:, 40)
    # lat is (:, 1)
    # pressure is (:, 1)
    input_wind = tf.keras.Input(
        shape=(40, ),
        dtype=the_type,
        name='input_wind'
    )
    input_lat = tf.keras.Input(
        shape=(1, ),
        dtype=the_type,
        name='input_lat'
    )
    input_press = tf.keras.Input(
        shape=(1, ),
        dtype=the_type,
        name='input_press'
    )
    inputs = [input_wind, input_lat, input_press]
    # concatenate them into a single to feed to original davenet under
    # the hood
    concatted_input = tf.keras.layers.concatenate(
        inputs,
        dtype=the_type
    )
    prev_layer = BatchNormalization(
        dtype=the_type,
        epsilon=0.00001,  # to match the PyTorch BatchNorm1d default
        name='batchnormlayer'
    )(concatted_input)

    for i in range(1, 10, 2):  # 1, 3, 5, 7, 9
        prev_layer = tf.keras.layers.Dense(
            256,
            activation="relu",
            dtype=the_type,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            name=f"dense_{i}"
        )(prev_layer)

    prev_layer = tf.keras.layers.Dense(
        64,
        activation="relu",
        dtype=the_type,
        kernel_initializer=tf.keras.initializers.GlorotNormal(),
        name='dense_11')(prev_layer)

    output_layers = []
    for i in range(40):
        nested_prev_layer = prev_layer
        for size in [32]:
            nested_prev_layer = tf.keras.layers.Dense(
                size,
                activation="relu",
                dtype=the_type,
                kernel_initializer=tf.keras.initializers.GlorotNormal(),
                name=f"branches.{i}.0",
            )(nested_prev_layer)

        # These produce the single value outputs from each branch.
        output_layers.append(Dense(
            1,
            dtype=the_type,
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            name=f"branches.{i}.2",
        )(nested_prev_layer))

    # Concatenate them into one tensor.
    concatted_output = tf.keras.layers.concatenate(
        output_layers,
        dtype=the_type
    )

    # Normalization layer
    # davenet does
    #
    #   # Un-standardize
    #   Y *= self.stds
    #   Y += self.means
    # i.e. it's an inversion of a normalization layer, so we'll set
    # invert=True.

    # Oh dear, invert doesn't seem to persist after saving
    # (see https://github.com/keras-team/keras/issues/17556).
    # Note that Normalization layer has been in keras forever, at least 3 years
    # and no one noticed this.
    # We'll have to manually invert the mean and variance here.
    # Since the Normalization layer will do (x - m) / s
    # we need m = -mean / std and s = 1 / std, but layer stores variance
    # which is s^2 = 1 / std^2

    # oh but there's a stddev of zero in there which mucks things up.
    # Use np.nan_to_num.
    means = np.nan_to_num(
        - checkpoint['means'].to('cpu').numpy() /
        checkpoint['stds'].to('cpu').numpy()
    )

    variance = np.nan_to_num(np.reciprocal(
        checkpoint['stds'].to('cpu').numpy() ** 2))
    normalized_output = tf.keras.layers.Normalization(
        invert=False,  # see above
        dtype=the_type,
        mean=means,
        variance=variance,
    )(concatted_output)

    # Finally declare the model.
    model = tf.keras.Model(inputs=inputs, outputs=normalized_output)

    # Most of the options here only apply to training, and are copied from
    # Learning-with-GWD-with-MIMA.  We do need to compile the Model before we
    # save it.
    adam_optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        amsgrad=False,
        clipvalue=.1,
    )
    model.compile(
            optimizer=adam_optimizer,
            loss=tf.keras.losses.LogCosh(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                name="log_cosh"
            ),
            metrics=[
                # Fits to Median: robust to unwanted outliers
                tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error",
                                                   dtype=None),
                # # Fits to Mean: robust to wanted outliers
                tf.keras.metrics.MeanSquaredError(name="mean_squared_error",
                                                  dtype=None),
                # # Twice diferentiable, combination of MSE and MAE
                tf.keras.metrics.LogCoshError(name="logcosh", dtype=None),
                # # STD of residuals
                tf.keras.metrics.RootMeanSquaredError(
                    name="root_mean_squared_error", dtype=None
                )
            ]
        )

    # Now take the weights from the PyTorch checkpoint and populate our model.
    # The mapping was done by manual inspect of the two models.

    # shared.0 is the py.BatchNorm1d layer -> tf.BatchNormalization
    chk = checkpoint['weights']
    bnweights = [w.to('cpu').numpy().T for w in (
        chk['shared.0.weight'],
        chk['shared.0.bias'],
        chk['shared.0.running_mean'],
        chk['shared.0.running_var']
    )]
    # model.layers[0] is the tf.InputLayer, there is no PyTorch equiv.
    model.get_layer('batchnormlayer').set_weights(bnweights)

    # shared.{1,3,5,7,9,11} are the shared Linear layers -> tf.Dense
    # shared.{2,4,6,8,10,12} are the ReLU activations.  These are not separate
    # layers in TF.
    for i in range(1, 12, 2):
        weights = [w.to('cpu').numpy().T for w in (
            chk[f"shared.{i}.weight"],
            chk[f"shared.{i}.bias"]
        )]
        model.get_layer(f"dense_{i}").set_weights(weights)

    # branch layers Linear -> tf.Dense
    for i in range(40):  # loop over branches
        for j in (0, 2):  # layer in branch
            weights = [w.to('cpu').numpy().T for w in (
                chk[f"branches.{i}.{j}.weight"],
                chk[f"branches.{i}.{j}.bias"]
            )]
            model.get_layer(f"branches.{i}.{j}").set_weights(weights)

    # Finish
    model.summary()
    model.save(args.output_dir)


if __name__ == '__main__':
    main()
