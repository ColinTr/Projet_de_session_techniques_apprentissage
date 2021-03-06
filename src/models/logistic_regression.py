import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from src.models.base_classifier import BaseClassifier


class MyLogisticRegression(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, max_iterations=1000, penalty='l2', is_dual=False, c=0.001):
        super().__init__(x_train, t_train, x_test, t_test, 4)
        self.max_iterations = max_iterations
        self.penalty = penalty
        self.dual = is_dual
        self.c = c
        self.classifier = LogisticRegression(max_iter=self.max_iterations, C=self.c, penalty=self.penalty,
                                             dual=self.dual, n_jobs=-1)

    def sklearn_random_grid_search(self, n_iter):
        print("======= Starting logistic regression grid search ========")
        distributions = dict(C=np.linspace(0.01, 1, 100))

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter, cv=5)

        search = random_search.fit(self.x_train, self.t_train)

        best_c = search.best_params_['C']

        print("Grid Search final hyper-parameters :\n"
              "     best_c=", best_c)

        return best_c

    def grid_search(self):
        print("======= Starting logistic regression grid search ========")
        best_accuracy = 0
        best_c = None

        for c in np.linspace(0.01, 5, 20):
            self.c = c

            self.classifier = LogisticRegression(max_iter=self.max_iterations, C=self.c, penalty=self.penalty,
                                                 dual=self.dual, n_jobs=-1)
            mean_cross_validation_accuracy = self.cross_validation()

            if mean_cross_validation_accuracy == 100:
                print("All train data was correctly classified during cross-validation !")

            if mean_cross_validation_accuracy > best_accuracy:
                best_accuracy = mean_cross_validation_accuracy
                best_c = self.c

        print("Grid Search final hyper-parameters :\n"
              "     best_c=", best_c)

        return best_c
