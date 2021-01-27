import numpy as np
import pytest

from src.errors import InvalidPaddingModeError
from src.layers.convolutional import ConvLayer2D, FastConvLayer2D, SuperFastConvLayer2D

if __name__ == "__main__":
    w = np.random.rand(5, 5, 3, 16)
    b = np.random.rand(16)
    conv_layer = ConvLayer2D(w, b)
    print(conv_layer)
