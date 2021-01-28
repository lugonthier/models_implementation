import numpy as np

from deepLearning.layer.Layer import Layer
from deepLearning.operation.Operation import Operation
from deepLearning.operation.specificOperation.WeightMultiply import WeightMultiply
from deepLearning.operation.specificOperation.BiasAdd import BiasAdd
from deepLearning.operation.specificOperation.Sigmoid import Sigmoid


class Dense(Layer):
    '''
    A fully connected layer that inherits from "Layer".
    '''
    def __init__(self, neurons: int, activation:Operation = Sigmoid()) -> None:
        '''
        Requires an activation function upon initalization.
        '''
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: np.ndarray) -> None:
        '''
        Defines the operations of a fully connected layer.
        '''
        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        #weights
        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        #bias
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]), BiasAdd(self.params[1]), self.activation]

        return None

