import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV

from src.models.base_classifier import BaseClassifier


class MyLinearDiscriminantAnalysis(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, solver='lsqr',
                 shrinkage=0):
        """Note : Shrinkage works only with lsqr and eigen solvers"""
        super().__init__(x_train, t_train, x_test, t_test, 4)
        self.solver = solver
        self.shrinkage = shrinkage
        self.classifier = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)

    def sklearn_random_grid_search(self, n_iter):
        print("======= Starting Linear discriminant grid search ========")
        distributions = dict(shrinkage=np.linspace(0.0, 1, 100))

        random_search = RandomizedSearchCV(self.classifier, distributions, n_jobs=-1, n_iter=n_iter, cv=5)

        search = random_search.fit(self.x_train, self.t_train)

        best_shrinkage = search.best_params_['shrinkage']

        print("Grid Search final hyper-parameters :\n"
              "     best_shrinkage=", best_shrinkage)

        return best_shrinkage

    def grid_search(self):
        if self.solver != 'lsqr' and self.solver != 'eigen':
            print("WARNING : In LinearDiscriminantAnalysis, if the solver is not lsqr or eigen, there is no "
                  "hyper-paramater to optimize")
            return

        print("======= Starting Linear discriminant grid search ========")
        best_accuracy = 0
        best_shrinkage = None

        for shrinkage in np.linspace(0.0, 1, 20):
            self.shrinkage = shrinkage

            self.classifier = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)
            mean_cross_validation_accuracy = self.cross_validation()

            if mean_cross_validation_accuracy == 100:
                print("All train data was correctly classified during cross-validation !")

            if mean_cross_validation_accuracy > best_accuracy:
                best_accuracy = mean_cross_validation_accuracy
                best_shrinkage = self.shrinkage

        print("Grid Search final hyper-parameters :\n"
              "     shrinkage=", best_shrinkage)

        return best_shrinkage
