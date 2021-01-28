import numpy as np

from deepLearning.utils.utils import assert_same_shape


class Operation(object):
    '''
    Base class for an *operation* in a neural network.
    '''

    def __init__(self):
        pass

    def forward(self, input_: np.ndarray):
        '''
        Stores input in the self._input instance variable
        Calls the self._output() function.
        '''
        self.input_ = input_
        self.output = self._output()

        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
       Calls the self.input_grad() function.
       Checks that the appropriate shapes match.
        '''
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self) -> np.ndarray:
        '''
        The _output method must be defined for each Operation
        '''
        raise NotImplementedError()

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        The _input_grad method must be defined for each Operation
        '''

        raise NotImplementedError()
