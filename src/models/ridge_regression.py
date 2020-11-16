import numpy as np

from sklearn.linear_model import RidgeClassifier
from src.models.base_classifier import BaseClassifier


class MyRidgeRegression(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, max_iterations=1000, lamb=1.0):
        super().__init__(x_train, t_train, x_test, t_test, 6)
        self.max_iterations = max_iterations
        self.lamb = lamb
        self.classifier = RidgeClassifier(max_iter=self.max_iterations, alpha=self.lamb)

    def grid_search(self):
        print("======= Starting Ridge regression grid search ========")
        best_accuracy = 0
        best_lamb = None

        for lamb in np.linspace(0.000000001, 2, 10):
            self.lamb = lamb

            self.classifier = RidgeClassifier(max_iter=self.max_iterations, alpha=self.lamb)

            mean_cross_validation_accuracy = self.cross_validation()

            if mean_cross_validation_accuracy == 100:
                print("All train data was correctly classified during cross-validation !")

            if mean_cross_validation_accuracy > best_accuracy:
                best_accuracy = mean_cross_validation_accuracy
                best_lamb = self.lamb

        return best_lamb
