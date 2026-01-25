import os
import numpy as np
import torch

from algs.train import train
from algs.unet import run_unet



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # run_unet()
    device = torch.device("cuda:0" if torch.cuda.is_available() and False else "cpu")
    imdir = './data/MouseData/'

    # Run training for multiple seeds and algorithms
    for seed in range(2):
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        # Trian Attention Autoencoder DeepMapper
        train(imdir, seed, alg="AttAE", num_epochs=200)
        # Trian RNN Autoencoder DeepMapper
        train(imdir, seed, alg="RNNAE", num_epochs=200)
