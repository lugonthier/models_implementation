from typing import Tuple

import numpy as np
import torch


def assert_same_shape(output: np.ndarray, output_grad: np.ndarray):
    assert output.shape == output_grad.shape, \
        '''
        Two tensors should have the same shape;
        instead, first Tensor's shape is {0}
        and second Tensor's shape is {1}.
        '''.format(tuple(output_grad.shape), tuple(output.shape))
    return None

def permute_data(X: torch.Tensor, y: torch.Tensor, seed=1) -> Tuple[torch.Tensor]:
    perm = torch.randperm(X.shape[0])
    return X[perm], y[perm]