"""
This python script outlines the forpy commands that will be done in
cg_drag_lmy.f90 In the correct folder, place module named model_arch
and the model weights h5 file. This additionally removes the bias
w.r.t. the shear metric estimated on the training data.
"""
# Load architecture.
# import torch
# import model_arch as m;
# n_d=[128,64,32,1]; n_in=40; n_out=33
# device = "cpu"
# model=m.model(n_d, n_in, n_out).to(device);
# for branch in model.branches: branch.to(device)

# # Load weights and set to evaluation mode.
# best_model = torch.load(f"model_weights.h5",map_location=torch.device('cpu'))
# model.load_state_dict(best_model['weights'])
# model.eval()

import numpy as np
# from numpy import float32, float64, max, min, zeros, array, searchsorted, inf, nan
import pathlib
from torch import from_numpy, no_grad, reshape, transpose, double
from torch import load, device
# Imports required when outputting to MiMA console:
# from pickle import dump
# from os.path import exists


# Initialize everything
def initialize():
    # Set to run on CPU
    device_str = "cpu"

    # Import the model architecture from adjacent file
    # import model_arch_ae09_rmbias as m;
    import model_arch_ae09 as m

    # Import model weights from .h5 file:
    dirpath = pathlib.Path(__file__).resolve().parent
    best_model = load(
        dirpath.joinpath("model_weights_ae98.h5"),
        map_location=device(device_str)
    )
    model = m.model(best_model).to(device_str)

    # Load weights and set to evaluation mode:
    model.load_state_dict(best_model["weights"])
    model.eval()
    # print("Model has been loaded with trained weights.")

    return model


# Compute drag
def compute_reshape_drag(*args):
    model, X3, X2, Y_out, num_col = args

    # Provide output to shell when running MiMA for debugging purposes:
    # num = 1
    # while exists(f"from_mima_{num:02d}")==True:
    #    num+=1
    # dump({'X3':X3,'X2':X2,'Y_out':Y_out},open(f"from_mima_{num:02d}",'wb'))
    # X2 = X2.astype(float32)
    # whichlat=zeros(X2.shape[0],dtype=int)
    # for j,la in enumerate(model.lat.astype(float32)):
    #    whichlat[X2[:,0]==la]=j
    # u_range = array([max(X3[l,0,:model.source_level[whichlat[l]]])-min(X3[l,0,:model.source_level[whichlat[l]]]) for l in range(X3.shape[0])])
    # del whichlat
    # i = searchsorted(model.bin_edges, u_range, side='left')
    # del u_range;

    # Reshape to dimensions required by model and convert to torch tensors:
    imax = 128
    for j in range(X2.shape[0] // imax):
        X2[j * imax: (j + 1) * imax, 1] = model.lon[:]
    X3 = from_numpy(X3.astype(np.float32))
    X2 = from_numpy(X2.astype(np.float32))

    # Run model and reshape output into form expected by MiMA:
    with no_grad():
        temp = model([X3, X2])

    # Remove mean bias of the corresponding sample urange bin statistic
    # temp =temp.double()- model.sm_mean[i,:]

    temp = reshape(temp, (num_col, imax, 40))
    temp = transpose(temp, 0, 1)
    Y_out[:, :, :] = temp.detach().numpy()[:, :, :]
    # del temp, i

    # Provide output to shell when running MiMA for debugging purposes:
    # dump({'Y_out':Y_out},open(f"to_mima_{num:02d}",'wb'))

    return Y_out


if __name__ == "__main__":
    # Short script for standaline testing and debugging from command line
    trained_model = initialize()
    X3 = np.asfortranarray(np.random.randn(128 * 2, 4, 40).astype(np.float32))
    X2 = np.asfortranarray(np.random.randn(128 * 2, 3).astype(np.float32))
    temp = np.random.randn(128 * 2, 40)
    Y_out = np.asfortranarray(np.zeros((128, 2, 40)), dtype=np.float64)
    result = compute_reshape_drag(trained_model, X3, X2, Y_out, 2)
    print(result)
