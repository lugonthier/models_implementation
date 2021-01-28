from deepLearning.operation.Operation import Operation
import numpy as np
from deepLearning.utils.utils import assert_same_shape


class ParamOperation(Operation):
    '''
    An Operation with parameters.
    '''

    def __init__(self, param: np.ndarray) -> np.ndarray:
        super().__init__()
        self.param = param

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        '''
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad )
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Every subclass of ParamOperation must implement _parap_grad.
        '''
        raise NotImplementedError()

