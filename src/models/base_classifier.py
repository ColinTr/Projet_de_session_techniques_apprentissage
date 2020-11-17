from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA


class BaseClassifier:
    def __init__(self, x_train, t_train, x_test, t_test, k_folds):
        self.classifier = None
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.k_folds = k_folds
        self.train_predictions = None

    def training(self):
        self.classifier.fit(self.x_train, self.t_train)
        return

    def prediction(self):
        self.train_predictions = self.classifier.predict(self.x_test)
        return

    def cross_validation(self):
        skf = StratifiedKFold(n_splits=self.k_folds)
        scores = cross_val_score(self.classifier, self.x_train, self.t_train, cv=skf)
        mean_score = scores.sum() / self.k_folds
        return mean_score
