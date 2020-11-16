import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from src.models.base_classifier import BaseClassifier


class MyAdaboostClassifier(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, n_estimators=50, learning_rate=1, base_estimator='None'):
        super().__init__(x_train, t_train, x_test, t_test, 4)
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.classifier = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                             base_estimator=self.base_estimator)

    def grid_search(self):
        print("============= Starting AdaBoost grid search =============")
        best_accuracy = 0
        best_learning_rate = None
        best_n_estimators = None
        best_base_estimator = None

        for base_estimator in [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]:
            self.base_estimator = base_estimator

            for learning_rate_i in np.linspace(0.5, 5, 5):
                self.learning_rate = learning_rate_i

                for n_estimators_i in np.linspace(50, 500, 10, dtype=np.int16):
                    self.n_estimators = n_estimators_i

                    self.classifier = AdaBoostClassifier(n_estimators=self.n_estimators,
                                                         learning_rate=self.learning_rate,
                                                         base_estimator=self.base_estimator)
                    mean_cross_validation_accuracy = self.cross_validation()

                    if mean_cross_validation_accuracy == 100:
                        print("All train data was correctly classified")

                    print(mean_cross_validation_accuracy)

                    if mean_cross_validation_accuracy > best_accuracy:
                        best_accuracy = mean_cross_validation_accuracy
                        best_learning_rate = self.learning_rate
                        best_n_estimators = self.n_estimators
                        best_base_estimator = self.base_estimator

        return best_learning_rate, best_n_estimators, best_base_estimator
