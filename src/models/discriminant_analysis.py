import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.models.base_classifier import BaseClassifier


class MyDiscriminantAnalysis(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test, reg_param=0.0, solver='lsqr',
                 shrinkage='None'):
        """Note : Shrinkage works only with lsqr and eigen solvers"""
        super().__init__(x_train, t_train, x_test, t_test, 5)
        self.solver = solver
        self.shrinkage = shrinkage
        self.reg_param = reg_param
        self.classifier = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)

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

        return best_shrinkage
