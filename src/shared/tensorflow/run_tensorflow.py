"""
Contains all python commands MiMA will use.

This module will use the TensorFlow model.

"""
import tensorflow as tf


# Initialize everything
def initialize(*args, **kwargs):
    """
    Initialize a WaveNet model and load weights.

    Parameters
    __________
    ignores all parameters

    """

    model = tf.keras.models.load_model('/home/sjc306/hpc-work/Datawave/MiMA/src/shared/tensorflow/saved_model')

    return model


# Compute drag
def compute_reshape_drag(*args):
    """
    Compute the drag from inputs using a neural net.

    Takes in input arguments passed from MiMA and outputs drag in shape desired by MiMA.
    Reshaping is performed in the calling Fortran.

    Parameters
    __________
    model :
        WaveNet model ready to be deployed.
    wind :
        U or V (128, num_col, 40)
    lat :
        latitudes (num_col)
    p_surf :
        surface pressure (128, num_col)
    Y_out :
        output prellocated in MiMA (128, num_col, 40)
    num_col :
        # of latitudes on this proc

    Returns
    -------
    Y_out :
        Results to be returned to MiMA
    """
    model, wind, lat, p_surf, Y_out, num_col = args

    # Make tensors
    # TF will make Tensors itself?

    # Apply model.
    temp = model.predict([wind, p_surf, lat], verbose=0)

    # Place in output array for MiMA.
    Y_out[:, :] = temp

    return Y_out
