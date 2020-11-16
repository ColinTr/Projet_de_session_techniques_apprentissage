import numpy as np

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.models.base_classifier import BaseClassifier


class MyRidgeRegression(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, max_iterations=1000, lamb=1.0):
        super().__init__(x_train, t_train, x_test, t_test, 6)
        self.max_iterations = max_iterations
        self.lamb = lamb
        self.classifier = RidgeClassifier(max_iter=self.max_iterations, alpha=self.lamb)

    def sklearn_random_grid_search(self, n_iter):
        print("========= Starting Ridge regression grid search =========")
        distributions = dict(alpha=np.linspace(0.000000001, 2, 100))

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter)

        search = random_search.fit(self.x_train, self.t_train)

        best_lamb = search.best_params_['alpha']

        print("Grid Search final hyper-parameters :\n"
              "     best_learning_rate=", best_lamb)

        return best_lamb

    def grid_search(self):
        print("========= Starting Ridge regression grid search =========")
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

        print("Grid Search final hyper-parameters :\n"
              "     best_learning_rate=", best_lamb)

        return best_lamb
