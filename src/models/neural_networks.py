import numpy as np

from sklearn.neural_network import MLPClassifier
from src.models.base_classifier import BaseClassifier


class MyNeuralNetwork(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, hidden_layer_sizes=(100, 100), lamb=0.0001, max_iter=200,
                 solver='adam', learning_rate='constant', activation='relu'):
        super().__init__(x_train, t_train, x_test, t_test, 5)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.max_iter = max_iter
        self.solver = solver
        self.lamb = lamb
        self.classifier = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                                        solver=self.solver, alpha=self.lamb, learning_rate=self.learning_rate,
                                        max_iter=self.max_iter)

    def grid_search(self):
        print("=========== Starting neural network grid search =========")
        best_accuracy = 0
        best_hidden_layer_sizes = None
        best_lamb = None

        # We first try with a single hidden layer
        for hidden_layer_size in np.linspace(50, 150, 5, dtype=np.int16):
            best_hidden_layer_sizes, best_lamb, best_accuracy = \
                self.grid_search_lambda(hidden_layer_size, best_accuracy, best_hidden_layer_sizes, best_lamb)
            print(self.hidden_layer_sizes)

        # Then we try with two hidden layers
        for hidden_layer_sizes_i in np.linspace(50, 150, 5, dtype=np.int16):
            for hidden_layer_sizes_j in np.linspace(50, 150, 5, dtype=np.int16):
                hidden_layer_sizes = (hidden_layer_sizes_i, hidden_layer_sizes_j)
                best_hidden_layer_sizes, best_lamb, best_accuracy = \
                    self.grid_search_lambda(hidden_layer_sizes, best_accuracy, best_hidden_layer_sizes, best_lamb)
                print(self.hidden_layer_sizes)

        return best_lamb, best_hidden_layer_sizes

    def grid_search_lambda(self, input_hidden_layer_sizes, input_best_accuracy, input_best_hidden_layer_sizes,
                           input_best_lamb):
        self.hidden_layer_sizes = input_hidden_layer_sizes

        output_best_accuracy = input_best_accuracy
        output_best_hidden_layer_sizes = input_best_hidden_layer_sizes
        output_best_lamb = input_best_lamb

        for lamb_i in np.linspace(0.000000001, 0.1, 5):
            self.lamb = lamb_i
            self.classifier = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter,
                                            activation=self.activation, solver=self.solver, alpha=self.lamb,
                                            learning_rate=self.learning_rate)

            mean_cross_validation_accuracy = self.cross_validation()

            if mean_cross_validation_accuracy == 100:
                print("All train data was correctly classified during cross-validation !")

            if mean_cross_validation_accuracy > input_best_accuracy:
                output_best_accuracy = mean_cross_validation_accuracy
                output_best_hidden_layer_sizes = self.hidden_layer_sizes
                output_best_lamb = self.lamb

        return output_best_hidden_layer_sizes, output_best_lamb, output_best_accuracy
