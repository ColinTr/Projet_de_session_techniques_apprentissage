import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.model_selection import RandomizedSearchCV

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

    def sklearn_random_grid_search(self, n_iter):
        print("============= Starting perceptron grid search ===========")
        distributions = dict(alpha=np.linspace(0.000000001, 2, 10),
                             eta0=np.linspace(0.0001, 1, 10))

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter, cv=5)

        search = random_search.fit(self.x_train, self.t_train)

        best_lamb = search.best_params_['alpha']
        best_eta0 = search.best_params_['eta0']

        print("Grid Search final hyper-parameters :\n"
              "     best_lamb=", best_lamb, "\n" +
              "     best_eta0=", best_eta0)

        return best_lamb, best_eta0

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
                    print("All train data was correctly classified during cross-validation !")

                if mean_cross_validation_accuracy > best_accuracy:
                    best_accuracy = mean_cross_validation_accuracy
                    best_lamb = self.lamb
                    best_eta0 = self.eta0

        print("Grid Search final hyper-parameters :\n"
              "     best_lamb=", best_lamb, "\n" +
              "     best_eta0=", best_eta0)

        return best_lamb, best_eta0
