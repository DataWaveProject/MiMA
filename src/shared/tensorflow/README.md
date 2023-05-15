A script that takes the "checkpoint" file created when the code in `../pytorch`
is run and generates a corresponding TensorFlow model.  Note that the script
has hard-coded the model layout and structure, so any changes to DaveNet will
have to be propagated here.

The script then takes the weights and other parameters from the checkpoint file
and applys them to the appropriate places in the TF DaveNet.  The resulting
model produces output that is the same as the PyTorch model to within machine
precision.

```
Usage:
# Construct an environment, populate with pip -r requirements.txt
python construct-tf-davenet.py <path-to-checkpoint-file> [-o output_dir]
```
The checkpoint file is the one called `network_wst.pkl` in the pytorch directory.
The output directory is where the TF model will be saved.  Defaults to
`saved_model`.
