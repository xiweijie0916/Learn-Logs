import numpy as np
import torch
import torch.nn.functional as F
from SoftMax import softMax1d, softMax2d


if __name__ == "__main__":
    z = np.array([1.0, 2.0, 3.0])
    print(softMax1d(z))
    print(F.softmax(torch.from_numpy(z)))

    z = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    print(softMax2d(z))
    print(F.softmax(torch.from_numpy(z)))