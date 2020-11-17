from sklearn.naive_bayes import GaussianNB
from src.models.base_classifier import BaseClassifier


class MyNaiveBayes(BaseClassifier):
    def __init__(self, x_train, t_train, x_test, t_test):
        super().__init__(x_train, t_train, x_test, t_test, 5)
        self.classifier = GaussianNB()

