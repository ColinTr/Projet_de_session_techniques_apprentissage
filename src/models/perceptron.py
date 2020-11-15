import numpy as np

from sklearn.linear_model import Perceptron
from src.models.base_classifier import BaseClassifier


class MyPerceptron(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, lamb=0.0001, max_iterations=1000, penalty='None', eta0=1):
        super().__init__(x_train, t_train, x_test, t_test, 6)
        self.max_iterations = max_iterations
        self.penalty = penalty
        self.eta0 = eta0
        self.lamb = lamb
        self.classifier = Perceptron(max_iter=self.max_iterations, alpha=self.lamb, penalty=self.penalty,
                                     eta0=self.eta0, n_jobs=-1)

    def grid_search(self):
        print("============ Starting perceptron grid search ============")
        best_accuracy = 0
        best_lamb = None
        best_eta0 = None

        for lamb_i in np.linspace(0.000000001, 2, 10):
            self.lamb = lamb_i

            for eta0_i in np.linspace(0.0001, 1, 10):
                self.eta0 = eta0_i

                self.classifier = Perceptron(max_iter=self.max_iterations, alpha=self.lamb, penalty=self.penalty,
                                             eta0=self.eta0, n_jobs=-1)
                mean_cross_validation_accuracy = self.cross_validation()

                if mean_cross_validation_accuracy == 100:
                    print("All train data was correctly classified")

                if mean_cross_validation_accuracy > best_accuracy:
                    best_accuracy = mean_cross_validation_accuracy
                    best_lamb = self.lamb
                    best_eta0 = self.eta0

        return best_lamb, best_eta0
