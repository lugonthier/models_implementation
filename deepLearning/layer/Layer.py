from typing import List
import numpy as np

from deepLearning.operation.Operation import Operation
from deepLearning.operation.ParamOperation import ParamOperation
from deepLearning.utils.utils import assert_same_shape


class Layer(object):
    '''
    A "layer" of neurons in a neural network.
    '''

    def __init__(self, neurons: int):
        '''
        The number of "neurons" roughly corresponds to the "breadth" of the layer
        '''
        self.neurons = neurons
        self.first = True
        self.params: List[np.ndarray] = []
        self.param_grads: List[np.ndarray] = []
        self.operations: List[Operation] = []

    def _setup_layer(self, num_in: int) -> None:
        '''
        The _setup_layer function must be implemented for each layer.
        '''
        raise NotImplementedError()

    def forward(self, input_: np.ndarray) -> np.ndarray:
        '''
        Passes input forward through a series of operations.
        '''
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_
        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Passes output_grad backward through a series of operations.
        Checks appropriate shapes.
        '''
        assert_same_shape(self.output, output_grad)
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)

        input_grad = output_grad
        self._param_grads()
        return input_grad

    def _param_grads(self) -> np.ndarray:
        '''
        Extracts the _param_grads from a layer's operations.
        '''

        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> np.ndarray:
        '''
        Exctracts the _params from a layer's operations.
        '''
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)
