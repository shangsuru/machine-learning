import numpy as np
from matplotlib import pyplot as plt

"""
# Backpropagation
dL = (-error)@np.reshape(np.insert(a[-1], 0, 1), (1, -1))
self.weights[-1] += learning_rate * dL

for i in range(1, len(self.weights) - 1):
    prod = None
    for k in range(i, len(self.weights) - 1):
        val = (
            self.weights[k+1] @
            np.insert(self.activation_func_df[k](z[k]), 0, 1)
            )

        prod = val if prod is None else prod @ val

    dL = - (
        (np.reshape(prod, (-1, 1)) @ error) @
        np.reshape(np.insert(a[i], 0, 1), (1, -1))
        )

    self.weights[i] += learning_rate * dL
"""

class NeuralNetworkold:
    """Represents a neural network.

    Args:
        layers: An array of numbers specifying the neurons per layer.
        activation_func: An array of activation functions corresponding to
            each layer of the neural network except the first.
        actionvation_func_df: An array of the derivatives of the respective
            activation functions.
    Attributes:
        activation_func: The activation functions for each layer of the network.
        weights: A numpy array of weights.
    """

    def __init__(self, layers, activation_func, activation_func_df, penalty_func, penalty_func_df, feature_func):
        self.dim = layers
        self.activation_func = activation_func
        self.activation_func_df = activation_func_df
        self.penalty_func = penalty_func
        self.penalty_func_df = penalty_func_df
        self.feature_func = feature_func
        self.weights = [np.random.random((layers[i + 1], layers[i] + 1)) for i in range(len(layers) - 1)]


    def predict(self, x):
        """Predicts the value of a data point.

        Args:
            X: The input vector to predict an output value for.
        Returns:
            The predicted value(s).
        """
        a = x
        z = None

        for i, weight in enumerate(self.weights[:-1]):
            z = weight@np.insert(a, 0, 1)
            a = self.activation_func[i](z)

        y = self.weights[-1]@np.insert(a, 0, 1)

        return np.argsort(y)[-1]

    def train(self, x, y, learning_rate=.1):
        """Trains the neural network on a single data point.

        Args:
            X: The input vector.
            Y: The output value corresponding to the input vectors.
        """

        # Forwardpropagation: build up prediction matrix
        z = []
        a = [x]
        for i, weight in enumerate(self.weights[:-1]):
            z.append(weight@self.feature_func(a[-1]))
            a.append(self.activation_func[i](z[-1]))

        pred_y = (self.weights[-1]@self.feature_func(a[-1])).reshape(-1, 1)

        K = len(self.weights)
        d = [None] * K
        d[-1] = (pred_y - y).reshape(-1, 1)@a[-1].reshape(1, -1)

        for i in range(1, K):
            k = K - i - 1
            d[k] = d[k+1]@(self.weights[k + 1]@np.append(self.activation_func_df[k](z[k]),1))
        
        for i in range(K):
            self.weights[i] -= learning_rate * (d[i]@x[i])



class NeuralNetwork:
    """Represents a neural network.

    Args:
        layers: An array of numbers specifying the neurons per layer.
        activation_func: An array of activation functions corresponding to
            each layer of the neural network except the first.
        actionvation_func_df: An array of the derivatives of the respective
            activation functions.
    Attributes:
        activation_func: The activation functions for each layer of the network.
        weights: A numpy array of weights.
    """

    def __init__(self, layers, activation_func, activation_func_df, penalty_func, penalty_func_df, feature_func):
        self.dim = layers
        self.activation_func = activation_func
        self.activation_func_df = activation_func_df
        self.penalty_func = penalty_func
        self.penalty_func_df = penalty_func_df
        self.feature_func = feature_func
        self.weights = [np.random.random((layers[i + 1], layers[i])) * 1e-4 for i in range(len(layers) - 1)]
        self.bias = [np.random.random((layers[i + 1],)) * 1e-4 for i in range(len(layers) - 1)]


    def predict(self, x):
        """Predicts the value of a data point.

        Args:
            X: The input vector to predict an output value for.
        Returns:
            The predicted value(s).
        """
        a = x
        z = None

        for i, weight in enumerate(self.weights):
            z = weight@a + self.bias[i]
            a = self.activation_func[i](z)

        y = a

        return np.argsort(y)[-1]


    

    def train(self, x, y, learning_rate=.01):

        """Trains the neural network on a single data point.

        Args:
            X: The input vector.
            Y: The output value corresponding to the input vectors.
        """
        # forward propagate
        def fp():
            h = [x]
            a = []

            for i in range(len(self.weights)):
                a.append(self.bias[i] + self.weights[i]@h[i])
                h.append(self.activation_func[i](a[i]))
            return h, a
        h, a = fp()
        pred_y = h[-1]

        g = self.penalty_func_df(pred_y, y).reshape(-1, 1)
        K = len(self.weights)
        delta_b = [None] * K
        delta_W = [None] * K
        for i in range(K):
            k = K - i - 1
            g = g * self.activation_func_df[k](a[k]).reshape(-1, 1)
            delta_b[k] = g
            delta_W[k] = g@h[k].reshape(1, -1)
            g = self.weights[k].T@g

        for i in range(K):
            self.weights[i] = self.weights[i] - learning_rate * delta_W[i]
            self.bias[i] = self.bias[i] - learning_rate * delta_b[i].reshape(-1)


    def gradientAtPoint(self, x, y):
        def fp():
            h = [x]
            a = []

            for i in range(len(self.weights)):
                a.append(self.bias[i] + self.weights[i]@h[i])
                h.append(self.activation_func[i](a[i]))
            return h, a
        h, a = fp()
        pred_y = h[-1]


        g = self.penalty_func_df(y, pred_y).reshape(-1, 1)
        K = len(self.weights)
        delta_b = [None] * K
        delta_W = [None] * K
        for i in range(K):
            k = K - i - 1
            g = g * self.activation_func_df[k](a[k]).reshape(-1, 1)
            delta_b[k] = g
            delta_W[k] = g@h[k].reshape(1, -1)
            g = self.weights[k].T@g

        return delta_W, delta_b


    def onehot(self, y):
        ret = [0] * self.dim[-1]
        ret[int(y)] = 1
        return np.array(ret).reshape(1,-1)


    def gradientDescent(self, X, Y, epoch=10, learning_rate=1):
        for j in range(epoch):
            print(j)
            dw_total = None
            db_total = None
            for i in range(len(X)):
                dw, db = self.gradientAtPoint(X[i], self.onehot(Y[i]))
                dw_total = dw if dw_total is None else dw_total + dw
                db_total = db if db_total is None else db_total + db
            
            
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - learning_rate * dw_total[i] / len(X)
                self.bias[i] = self.bias[i] - learning_rate * db_total[i].reshape(-1) / len(X)


def onehot(y):
    ret = [0] * 10
    ret[int(y)] = 1
    return np.array(ret).reshape(1,-1)


def error(nn, X, Y):
    error = 0
    for i, x in enumerate(X):
        y = Y[i]
        
        y_ = nn.predict(x)
        error += 1 if y != y_ else 0
    return error / len(X)

def main():
    # Prepare data
    X_train = np.loadtxt('./dataSets/mnist_small_train_in.txt', delimiter=",")
    Y_train = np.loadtxt('./dataSets/mnist_small_train_out.txt', delimiter=",")
    X_test = np.loadtxt('./dataSets/mnist_small_test_in.txt', delimiter=",")
    Y_test = np.loadtxt('./dataSets/mnist_small_test_out.txt', delimiter=",")

    D = X_train[0].shape[0]

    # Create and train neural network
    def act_func(x): return 1 / (1 + np.exp(-x))
    def act_func_df(x): return act_func(x) * (1- act_func(x))
    def pen_func(x, y): return .5*(x - y)**2
    def pen_func_df(x, y): return (x - y)
    ff = lambda x: np.append(x, 1)

    layers = [D, 200, 10]
    nn = NeuralNetwork(
        layers, [act_func] * (len(layers) - 1),
        [act_func_df] * (len(layers) - 1), pen_func, pen_func_df,
        ff
        )
    print("Pre-Training:")
    print("Training Error: {}".format(error(nn, X_train, Y_train)))
    print("Test Error: {}".format(error(nn, X_test, Y_test)))

    nn.gradientDescent(X_train, Y_train)


    print("Post-Training:")
    print("Training Error: {}".format(error(nn, X_train, Y_train)))
    print("Test Error: {}".format(error(nn, X_test, Y_test)))

if __name__ == "__main__":
    main()
