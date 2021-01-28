from deepLearning.operation.ParamOperation import ParamOperation
import numpy as np

class BiasAdd(ParamOperation):
    '''
    Compute bias addition.
    '''

    def __init__(self, B: np.ndarray):
        '''
        Initialize Operation with self.param = B.
        Check appropriate shape.
        '''
        assert B.shape[0] == 1
        super().__init__(B)

    def _param_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Compute parameter gradient.
        '''
        param_grad = np.ones_like(self.param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])

    def _input_grad(self, output_grad: np.ndarray) -> np.ndarray:
        '''
        Compute input gradient.
        '''
        return np.ones_like(self.input_) * output_grad

    def _output(self) -> np.ndarray:
        '''
        Compute output.
        '''
        return self.input_ + self.param
