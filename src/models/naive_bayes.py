from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from src.models.base_classifier import BaseClassifier
import numpy as np


class MyNaiveBayes(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, var_smoothing=1e-9):
        super().__init__(x_train, t_train, x_test, t_test, 4)
        self.smoothing = var_smoothing
        self.classifier = GaussianNB()

    def sklearn_random_grid_search(self, n_iter):
        print("======= Starting Gaussian Naive Bayes grid search =======")
        distributions = dict(var_smoothing=np.linspace(1e-9, 1, 1000))

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter)

        search = random_search.fit(self.x_train, self.t_train)

        best_smoothing = search.best_params_['var_smoothing']

        print("Grid Search final hyper-parameters :\n"
              "     best_smoothing=", best_smoothing)

        return best_smoothing
