from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from src.models.base_classifier import BaseClassifier
from sklearn.ensemble import IsolationForest
import numpy as np


class MyNaiveBayes(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, var_smoothing=1e-9):
        super().__init__(x_train, t_train, x_test, t_test, 4)
        self.var_smoothing_i = var_smoothing
        self.classifier = GaussianNB()
        self.original_x_train = x_train
        self.original_t_train = t_train

    def sklearn_random_grid_search(self, n_iter):
        print("======= Starting Gaussian Naive Bayes grid search =======")
        distributions = dict(var_smoothing=np.linspace(1e-9, 1, 1000))

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter)

        search = random_search.fit(self.x_train, self.t_train)

        best_smoothing = search.best_params_['var_smoothing']

        print("Grid Search final hyper-parameters :\n"
              "     best_smoothing=", best_smoothing)

        return best_smoothing

    def grid_search(self):
        print("============ Starting perceptron grid search ============")
        best_accuracy = 0
        best_var_smoothing = None
        best_contamination = None

        for var_smoothing_i in np.linspace(0.001, 1, 5):
            self.var_smoothing_i = var_smoothing_i

            for contamination_i in np.linspace(0, 0.5, 20):
                iso = IsolationForest(contamination=contamination_i)
                yhat = iso.fit_predict(self.original_x_train)
                mask = yhat != -1
                self.x_train, self.t_train = self.original_x_train[mask, :], self.original_t_train[mask]

                self.classifier = GaussianNB(var_smoothing=var_smoothing_i)
                mean_cross_validation_accuracy = self.cross_validation()

                if mean_cross_validation_accuracy == 100:
                    print("All train data was correctly classified during cross-validation !")

                if mean_cross_validation_accuracy > best_accuracy:
                    best_accuracy = mean_cross_validation_accuracy
                    best_var_smoothing = var_smoothing_i
                    best_contamination = contamination_i

        print("Grid Search final hyper-parameters :\n"
              "     best_var_smoothing=", best_var_smoothing, "\n" +
              "     best_contamination=", best_contamination)

        return best_var_smoothing, best_contamination
