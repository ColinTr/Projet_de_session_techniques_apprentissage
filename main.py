import sys

from sklearn.metrics import accuracy_score
from src.data.data_handler import DataHandler
from sklearn.model_selection import StratifiedShuffleSplit
from src.models.perceptron import MyPerceptron
from src.models.logistic_regression import MyLogisticRegression
from src.models.support_vector_machines import MySVM


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python data_handler.py train_data_input_filepath output_filepath\n")
        print(
            "Exemple (Windows) : python main.py data\\raw\\train\\leaf-classification-train.csv data\\processed\n")
        print("Exemple (Linux) : python main.py data/raw/train/leaf-classification-train.csv data/processed\n")

    else:
        data_input_filepath = sys.argv[1]
        output_filepath = sys.argv[2]

        print("=============== Reading and handling data ===============")
        dh = DataHandler(data_input_filepath, output_filepath)
        dh.main()
        data, data_normalized_centered, labels, species =\
            dh.read_all_output_files()

        # Let's create a train and test dataset
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
        # We take the first split of our sss
        train_index, test_index = next(sss.split(data_normalized_centered, labels))
        x_train, x_test = data_normalized_centered[train_index], data_normalized_centered[test_index]
        t_train, t_test = labels[train_index].T[0], labels[test_index].T[0]

        perceptron = MyPerceptron(x_train, t_train, x_test, t_test)
        best_lamb, best_eta0 = perceptron.grid_search()
        print("Grid Search final hyper-parameters :", best_lamb, ", ", best_eta0)
        perceptron = MyPerceptron(x_train, t_train, x_test, t_test, lamb=best_lamb, eta0=best_eta0)
        perceptron.training()
        print("Train accuracy : {:.4%}".format(perceptron.classifier.score(perceptron.x_train, perceptron.t_train)))
        perceptron.prediction()
        print("Test accuracy: {:.4%}".format(accuracy_score(perceptron.t_test, perceptron.train_predictions)))

        logistic_regression = MyLogisticRegression(x_train, t_train, x_test, t_test)
        best_c = logistic_regression.grid_search()
        print("Grid Search final hyper-parameters :", best_c)
        logistic_regression = MyLogisticRegression(x_train, t_train, x_test, t_test, c=best_c)
        logistic_regression.training()
        print("Train accuracy : {:.4%}".format(logistic_regression.classifier.score(logistic_regression.x_train,
                                                                                    logistic_regression.t_train)))
        logistic_regression.prediction()
        print("Test accuracy : {:.4%}".format(accuracy_score(logistic_regression.t_test, logistic_regression.
                                                             train_predictions)))

        svm = MySVM(x_train, t_train, x_test, t_test)
        best_c, best_gamma = svm.grid_search()
        print("Grid Search final hyper-parameters :", best_c, ", ", best_gamma)
        svm = MySVM(x_train, t_train, x_test, t_test, c=best_c, gamma=best_gamma)
        svm.training()
        print("Train accuracy : {:.4%}".format(svm.classifier.score(svm.x_train, svm.t_train)))
        svm.prediction()
        print("Test accuracy: {:.4%}".format(accuracy_score(svm.t_test, svm.train_predictions)))

        print("Done")
    return


if __name__ == '__main__':
    main()
