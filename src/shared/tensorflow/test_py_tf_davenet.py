import torch
import tensorflow as tf
import numpy as np
import sys

sys.path.append('../pytorch')

import arch_davenet as m

def main():
    # Load PyTorch model
    checkpoint = torch.load('../pytorch/network_wst.pkl',
                      map_location=torch.device('cpu'))
    pytmodel = m.WaveNet(checkpoint).to('cpu')

    # Load weights and set to evaluation mode.
    pytmodel.load_state_dict(checkpoint["weights"])
    pytmodel.eval()

    # Load TF model
    tfmodel = tf.keras.models.load_model('saved_model')

    # Generate random inputs
    # Guessing at magnitudes
    wind = np.random.randn(1,40) * 100
    lat = np.random.randn(1,1) * 6
    press = np.random.randn(1,1) * 10
    inps = [wind, lat, press]

    pyt_inps = [torch.tensor(nd, device='cpu') for nd in inps]

    # Run models, compare
    pyt_ans = pytmodel(*pyt_inps)
    tf_ans = tfmodel(inps)

    difference = tf_ans - pyt_ans.to('cpu').numpy(force=True)
    print(difference)

if __name__ == '__main__':
    main()
