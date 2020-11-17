import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.models.base_classifier import BaseClassifier


class MyAdaboostClassifier(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, base_estimator, n_estimators=50, learning_rate=1):
        super().__init__(x_train, t_train, x_test, t_test, 4)
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.classifier = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                             base_estimator=self.base_estimator)

    def sklearn_random_grid_search(self, n_iter):
        print("============= Starting AdaBoost grid search =============")
        distributions = dict(learning_rate=np.linspace(0.000001, 10, 20),
                             n_estimators=np.linspace(50, 250, 20, dtype=np.int16),
                             base_estimator=[DecisionTreeClassifier(),
                                             RandomForestClassifier(),
                                             ExtraTreesClassifier()])

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter, cv=5)

        search = random_search.fit(self.x_train, self.t_train)

        best_learning_rate = search.best_params_['learning_rate']
        best_n_estimators = search.best_params_['n_estimators']
        best_base_estimator = search.best_params_['base_estimator']

        print("Grid Search final hyper-parameters :\n"
              "     best_learning_rate=", best_learning_rate, "\n" +
              "     best_n_estimators=", best_n_estimators, "\n" +
              "     best_base_estimator=", best_base_estimator)

        return best_learning_rate, best_n_estimators, best_base_estimator

    def grid_search(self):
        print("============= Starting AdaBoost grid search =============")
        best_accuracy = 0
        best_learning_rate = None
        best_n_estimators = None
        best_base_estimator = None

        for base_estimator in [DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier()]:
            self.base_estimator = base_estimator

            for learning_rate_i in np.linspace(0.0001, 5, 10):
                self.learning_rate = learning_rate_i

                for n_estimators_i in np.linspace(50, 250, 5, dtype=np.int16):
                    self.n_estimators = n_estimators_i

                    self.classifier = AdaBoostClassifier(n_estimators=self.n_estimators,
                                                         learning_rate=self.learning_rate,
                                                         base_estimator=self.base_estimator)
                    mean_cross_validation_accuracy = self.cross_validation()

                    if mean_cross_validation_accuracy == 100:
                        print("All train data was correctly classified during cross-validation !")

                    if mean_cross_validation_accuracy > best_accuracy:
                        best_accuracy = mean_cross_validation_accuracy
                        best_learning_rate = self.learning_rate
                        best_n_estimators = self.n_estimators
                        best_base_estimator = self.base_estimator

        print("Grid Search final hyper-parameters :\n"
              "     best_learning_rate=", best_learning_rate, "\n" +
              "     best_n_estimators=", best_n_estimators, "\n" +
              "     best_base_estimator=", best_base_estimator)

        return best_learning_rate, best_n_estimators, best_base_estimator
