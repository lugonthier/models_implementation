from typing import Tuple

from deepLearning.neuralNetwork.NeuralNetwork import NeuralNetwork
from deepLearning.optimizer.Optimizer import Optimizer
import numpy as np
from deepLearning.utils.utils import permute_data


class Trainer(object):
    '''
    Trains a neural network
    '''
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        '''
        Requires a neural network and an optimizer in order for training to occur.
        Assign the neural network as an instance variable to the optimizer.
        '''
        self.optim = optim
        self.net = net
        setattr(self.optim, 'net', self.net )

    def generate_batches(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         size: int = 32) -> Tuple[np.ndarray]:
        '''
        Generates batches for training
        '''
        assert X.shape[0] == y.shape[0], \
            '''
            features and target must have the same number of rows, instead
            features has {0} and target has {1}
            '''.format(X.shape[0], y.shape[0])

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii:ii + size], y[ii:ii + size]

            yield X_batch, y_batch


    def fit(self, X_train: np.ndarray,y_train: np.ndarray,
            X_test: np.ndarray, y_test: np.ndarray, epochs: int=100,
            eval_every: int=10, batch_size: int=32, seed:int = 1, restart: bool = True) -> None:

        '''
        Fits the nn on the training data for a certain number of epochs.
        Every "eval_every" epochs, it evaluates the nn on the testing set.

        '''
        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True

        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)

            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):

                self.net.train_batch(X_batch, y_batch)
                self.optim.step()
            if(e+1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                print(f"Validation loss after {e+1} epochs is {loss:.3f}")

