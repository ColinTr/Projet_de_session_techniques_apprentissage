from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


class MyPerceptron:
    def __init__(self, x_train, t_train, x_test, t_test, lamb, max_iterations):
        self.classifier = Perceptron(eta0=0.001, max_iter=1000, penalty='l2')

        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.w = None
        self.w_0 = None
        self.lamb = lamb
        self.max_iterations = max_iterations

    def training(self):
        self.classifier.fit(self.x_train, self.t_train)
        self.w = self.classifier.coef_[0]
        self.w_0 = self.classifier.intercept_[0]

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')
        return

    def prediction(self):
        train_predictions = self.classifier.predict(self.x_test)
        acc = accuracy_score(self.t_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))
        return

    def cross_validation(self):
        return
