import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
from src.models.base_classifier import BaseClassifier


class MySVM(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, c=1.0, gamma='scale', kernel='rbf'):
        super().__init__(x_train, t_train, x_test, t_test, 6)
        self.kernel = kernel
        self.gamma = gamma
        self.c = c
        self.classifier = SVC(C=self.c, gamma=self.gamma, kernel=self.kernel)

    def sklearn_random_grid_search(self, n_iter):
        print("================ Starting SVM grid search ===============")
        distributions = dict(C=np.linspace(0.01, 1000, 10),
                             gamma=np.linspace(0.00001, 1, 10))

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter, cv=5)

        search = random_search.fit(self.x_train, self.t_train)

        best_c = search.best_params_['C']
        best_gamma = search.best_params_['gamma']

        print("Grid Search final hyper-parameters :\n"
              "     best_c=", best_c, "\n" +
              "     best_gamma=", best_gamma)

        return best_c, best_gamma

    def grid_search(self):
        print("================ Starting SVM grid search ===============")
        best_accuracy = 0
        best_c = None
        best_gamma = None

        for c_i in np.linspace(0.01, 1000, 20):
            self.c = c_i

            for gamma_i in np.linspace(0.00001, 1, 20):
                self.gamma = gamma_i

                self.classifier = SVC(C=self.c, gamma=self.gamma, kernel=self.kernel)

                mean_cross_validation_accuracy = self.cross_validation()

                if mean_cross_validation_accuracy == 100:
                    print("All train data was correctly classified during cross-validation !")

                if mean_cross_validation_accuracy > best_accuracy:
                    best_accuracy = mean_cross_validation_accuracy
                    best_c = self.c
                    best_gamma = self.gamma

        print("Grid Search final hyper-parameters :\n"
              "     best_c=", best_c, "\n"
              "     best_gamma=", best_gamma)

        return best_c, best_gamma
