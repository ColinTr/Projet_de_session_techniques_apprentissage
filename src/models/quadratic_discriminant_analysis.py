import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV

from src.models.base_classifier import BaseClassifier


class MyQuadraticDiscriminantAnalysis(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, reg_param=0.0, store_covariance=False):
        super().__init__(x_train, t_train, x_test, t_test, 4)
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.classifier = QuadraticDiscriminantAnalysis(reg_param=self.reg_param,
                                                        store_covariance=self.store_covariance)

    def sklearn_random_grid_search(self, n_iter):
        print("====== Starting Quadratic discriminant grid search ======")
        distributions = dict(reg_param=np.linspace(0.000001, 1, 50),
                             store_covariance=[True, False])

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter, cv=5)

        search = random_search.fit(self.x_train, self.t_train)

        best_reg_param = search.best_params_['reg_param']
        best_store_covariance = search.best_params_['store_covariance']

        print("Grid Search final hyper-parameters :\n"
              "     best_reg_param=", best_reg_param, "\n"
              "     best_store_covariance=", best_store_covariance)

        return best_reg_param, best_store_covariance

    def grid_search(self):
        print("====== Starting Quadratic discriminant grid search ======")
        best_accuracy = 0
        best_reg_param = None
        best_store_covariance = None

        for reg_param in np.linspace(0.000001, 1, 50):
            self.reg_param = reg_param

            for store_covariance in [True, False]:
                self.store_covariance = store_covariance

                self.classifier = QuadraticDiscriminantAnalysis(reg_param=self.reg_param,
                                                                store_covariance=self.store_covariance)
                mean_cross_validation_accuracy = self.cross_validation()

                if mean_cross_validation_accuracy == 100:
                    print("All train data was correctly classified during cross-validation !")

                if mean_cross_validation_accuracy > best_accuracy:
                    best_accuracy = mean_cross_validation_accuracy
                    best_reg_param = self.reg_param
                    best_store_covariance = self.store_covariance

        print("Grid Search final hyper-parameters :\n"
              "     best_reg_param=", best_reg_param, "\n"
              "     best_store_covariance=", best_store_covariance)

        return best_reg_param, best_store_covariance
