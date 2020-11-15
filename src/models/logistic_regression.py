import numpy as np

from sklearn.linear_model import LogisticRegression
from src.models.base_classifier import BaseClassifier


class MyLogisticRegression(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, lamb, max_iterations=1000, penalty='l2', is_dual=False, C=0.001):
        super().__init__(x_train, t_train, x_test, t_test, lamb, 6)
        self.dual = is_dual
        self.C = C
        self.max_iterations = max_iterations
        self.penalty = penalty
        self.classifier = LogisticRegression(max_iter=self.max_iterations, C=self.C, penalty=self.penalty,
                                             dual=self.dual, n_jobs=-1)

    def grid_search(self):
        best_accuracy = 0
        best_C = None

        for C in np.linspace(0.01, 1, 20):
            self.C = C

            self.classifier = LogisticRegression(max_iter=self.max_iterations, C=self.C, penalty=self.penalty,
                                                 dual=self.dual, n_jobs=-1)
            mean_cross_validation_accuracy = self.cross_validation()
            if mean_cross_validation_accuracy > best_accuracy:
                best_accuracy = mean_cross_validation_accuracy
                best_C = self.C

        self.training()
        self.prediction()
        print("Grid search final train accuracy : {:.4%}".format(self.classifier.score(self.x_train, self.t_train)))
        return best_C
