import torch
import numpy as np
from torch.autograd import Variable


# Don't actually need to store the data; just generate it on the fly
def sine_generator(seq_size=6000, mu=256):
    framerate = 44100
    t = np.linspace(0, 5, framerate * 5)
    data = np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 225 * t)
    data = data/2
    while True:
        start = np.random.randint(0, data.shape[0] - seq_size)
        ys = data[start:start + seq_size]
        ys = encode_mu_law(ys, mu)
        yield Variable(torch.from_numpy(ys[:seq_size]))


def encode_mu_law(x, mu=256):
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.long)


def decode_mu_law(y, mu=256):
    mu = mu - 1
    fx = (y - 0.5) / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return x
