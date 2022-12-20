# import torch
import torch.nn as nn
from torch import zeros, reshape, cat, max, min, tensor

# import torch.nn.functional as F
# import numpy as np
import sys

sys.path.append("/home/lmy7879/gwp/codes")
# from utils import count_params, init_xavier
from numpy import reshape, float32, array, searchsorted

# x3 is shaped (batch_size, 80).
# gu is shaped (batch_size, 40).
def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight)


class model(nn.Module):
    def __init__(self, s_dict, n_levels=40):
        super(model, self).__init__()
        self.n_levels = n_levels  # Input size. (length of column)
        self.n_ch = s_dict["n_ch"]
        self.n_d = s_dict["n_d"]
        self.kernel_size = 3
        self.n_dilations = s_dict["n_dilations"]
        self.lon = s_dict["lon"]
        self.lat = s_dict["lat"]
        self.n_flat_in = int(self.n_ch[3] * n_levels / 2) + 3
        self.n_flat_out = int(self.n_ch[3] * n_levels / 2)
        self.input_stats = {
            "mean_3": s_dict["input_3d_mean"],
            "std_3": s_dict["input_3d_std"],
            "mean_sp": s_dict["sp_mean"],
            "std_sp": s_dict["sp_std"],
        }
        self.output_stats = {"mean": s_dict["target_mean"], "std": s_dict["target_std"]}
        self.sm_mean = s_dict["sm_mean"]  # shear metric
        self.source_level = s_dict["source_level"]
        self.bin_edges = s_dict["bin_edges"]
        del s_dict
        self.cnn_encode = nn.Sequential(
            nn.Conv1d(
                self.n_ch[0],
                self.n_ch[1],
                self.kernel_size,
                1,
                padding="same",
                dilation=self.n_dilations[0],
            ),
            nn.ELU(),
            nn.Conv1d(
                self.n_ch[1],
                self.n_ch[2],
                self.kernel_size,
                1,
                padding="same",
                dilation=self.n_dilations[1],
            ),
            nn.ELU(),
            nn.Conv1d(
                self.n_ch[2],
                self.n_ch[3],
                self.kernel_size,
                1,
                padding="same",
                dilation=self.n_dilations[2],
            ),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
        )
        self.dense = nn.Sequential(
            nn.Linear(self.n_flat_in, self.n_d),
            nn.ELU(),
            nn.Linear(self.n_d, self.n_d),
            nn.ELU(),
            # nn.Linear(self.n_d,    self.n_d), nn.ELU(), nn.BatchNorm1d(100)
            # nn.Linear(self.n_d,    self.n_d), nn.ELU(), nn.BatchNorm1d(100)
            nn.Linear(self.n_d, self.n_flat_out),
            nn.ELU(),
        )
        self.cnn_decode = nn.Sequential(
            nn.ConvTranspose1d(
                self.n_ch[3],
                self.n_ch[2],
                self.kernel_size,
                2,
                padding=1,
                output_padding=1,
                dilation=self.n_dilations[3],
            ),
            nn.ELU(),
            nn.ConvTranspose1d(
                self.n_ch[2],
                self.n_ch[1],
                self.kernel_size,
                1,
                padding=1,
                dilation=self.n_dilations[4],
            ),
            nn.ELU(),
            nn.ConvTranspose1d(
                self.n_ch[1],
                1,
                self.kernel_size,
                1,
                padding=1,
                dilation=self.n_dilations[5],
            ),
        )
        self.cnn_encode.apply(init_xavier)
        self.dense.apply(init_xavier)
        self.cnn_decode.apply(init_xavier)

    def forward(self, x):
        # Unpack
        x3, xloc = x
        # print(x3.shape)
        # print(xloc.shape)
        # get latitude info
        batchsize = xloc.shape[0]
        whichlat = zeros(batchsize, dtype=int)
        for j, la in enumerate(self.lat.astype(float32)):
            whichlat[xloc[:, 0] == la] = j
        u_range = tensor(
            array(
                [
                    max(x3[l, 0, : self.source_level[whichlat[l]]])
                    - min(x3[l, 0, : self.source_level[whichlat[l]]])
                    for l in range(batchsize)
                ]
            )
        )
        del whichlat
        i = searchsorted(self.bin_edges, u_range, side="left")
        del u_range

        # Standardize
        x3 -= reshape(self.input_stats["mean_3"], (1, 4, 1))
        x3 /= reshape(self.input_stats["std_3"], (1, 4, 1))
        xloc[:, -1] -= self.input_stats["mean_sp"]
        xloc[:, -1] /= self.input_stats["std_sp"]
        # Encod:,e 3d variables
        z = self.cnn_encode(x3)  # ; print(z3.shape)

        # Concatenate with loc variables
        z = cat((z, xloc), axis=1)

        # Dense it up.
        z = self.dense(z)

        # Reshape for convolutions.
        z = reshape(z, (z.shape[0], self.n_ch[3], int(self.n_levels / 2)))

        # Decode.
        gu = self.cnn_decode(z).squeeze()

        # Unstandardize
        gu *= self.output_stats["std"]
        gu += self.output_stats["mean"]
        gu -= self.sm_mean[i, :].astype(float32)
        del i
        return gu
