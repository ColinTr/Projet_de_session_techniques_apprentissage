import numpy as np

from sklearn.svm import SVC
from src.models.base_classifier import BaseClassifier


class MySVM(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, c=1.0, gamma='scale', kernel='rbf'):
        super().__init__(x_train, t_train, x_test, t_test, 6)
        self.kernel = kernel
        self.gamma = gamma
        self.c = c
        self.classifier = SVC(C=self.c, gamma=self.gamma, kernel=self.kernel)

    def grid_search(self):
        print("================ Starting SVM grid search ===============")
        best_accuracy = 0
        best_c = None
        best_gamma = None

        for c_i in np.linspace(0.01, 1000, 10):
            self.c = c_i

            for gamma_i in np.linspace(0.00001, 1, 10):
                self.gamma = gamma_i

                self.classifier = SVC(C=self.c, gamma=self.gamma, kernel=self.kernel)

                mean_cross_validation_accuracy = self.cross_validation()

                if mean_cross_validation_accuracy == 100:
                    print("All train data was correctly classified during cross-validation !")

                if mean_cross_validation_accuracy > best_accuracy:
                    best_accuracy = mean_cross_validation_accuracy
                    best_c = self.c
                    best_gamma = self.gamma

        return best_c, best_gamma
