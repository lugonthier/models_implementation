import numpy as np
from deepLearning.layer.specificLayer.DenseLayer import Dense
from deepLearning.loss.specificLoss.MeanSquaredError import MeanSquaredError
from deepLearning.neuralNetwork.NeuralNetwork import NeuralNetwork
from deepLearning.operation.specificOperation.Linear import Linear
from deepLearning.operation.specificOperation.Sigmoid import Sigmoid
from deepLearning.optimizer.specificOptimizer.SGD import SGD
from deepLearning.trainer.Trainer import Trainer
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def to_2d_np(a: np.ndarray,
             type: str = "col") -> np.ndarray:
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
        "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


if __name__ == '__main__':
    # Charge Data
    boston = load_boston()
    data = boston.data
    target = boston.target
    features = boston.feature_names

    # Scaling the data
    s = StandardScaler()
    data = s.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

    # make target 2d array
    y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)

    linear_regression = NeuralNetwork(layers=[Dense(neurons=1, activation=Linear())], loss=MeanSquaredError(),
                                      seed=20190501)

    nn = NeuralNetwork(layers=[Dense(neurons=13, activation=Sigmoid()),
                               Dense(neurons=1, activation=Linear())],
                       loss=MeanSquaredError(),
                       seed=20190501)

    optimizer = SGD(lr=0.01)
    trainer = Trainer(linear_regression, optimizer)

    trainer.fit(X_train, y_train, X_test, y_test, epochs=50, eval_every=10, seed=20190501)
