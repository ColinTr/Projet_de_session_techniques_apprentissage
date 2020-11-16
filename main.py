import sys

from sklearn.metrics import accuracy_score
from src.data.data_handler import DataHandler
from sklearn.model_selection import StratifiedShuffleSplit

from src.models.discriminant_analysis import MyDiscriminantAnalysis
from src.models.neural_networks import MyNeuralNetwork
from src.models.perceptron import MyPerceptron
from src.models.logistic_regression import MyLogisticRegression
from src.models.support_vector_machines import MySVM
from src.models.ridge_regression import MyRidgeRegression


def main():
    if len(sys.argv) < 5:
        print("Usage: python data_handler.py train_data_input_filepath output_filepath classifier "
              "centered_normalized_data\n")
        print("classifier : 0=>all, 1=>ridge, 2=>discriminant analysis, 3=>logistic,"
              " 4=>neural networks, 5=>perceptron, 6=>SVM\n")
        print("centered_normalized_data : 0=>raw data, 1=>centered and normalized data\n")
        print("Exemple (Windows): python main.py data\\raw\\train\\leaf-classification-train.csv data\\processed 0 1\n")
        print("Exemple (Linux): python main.py data/raw/train/leaf-classification-train.csv data/processed 0 1\n")

    else:
        data_input_filepath = sys.argv[1]
        output_filepath = sys.argv[2]
        classifier = int(sys.argv[3])
        centered_normalized_bool = int(sys.argv[4])

        if centered_normalized_bool != 0 and centered_normalized_bool != 1:
            print("Incorrect value in centered_normalized_data parameter")
            return

        if classifier < 0 or classifier > 6:
            print("Incorrect value in classifier parameter")
            return

        print("=============== Reading and handling data ===============")
        dh = DataHandler(data_input_filepath, output_filepath)
        dh.main()
        raw_data, data_normalized_centered, labels, species = \
            dh.read_all_output_files()

        # Let's create a train and test dataset
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
        # We take the first split of our sss
        x_train = x_test = t_train = t_test = None
        if centered_normalized_bool == 0:
            train_index, test_index = next(sss.split(raw_data, labels))
            x_train, x_test = raw_data[train_index], raw_data[test_index]
            t_train, t_test = labels[train_index].T[0], labels[test_index].T[0]
        if centered_normalized_bool == 1:
            train_index, test_index = next(sss.split(data_normalized_centered, labels))
            x_train, x_test = data_normalized_centered[train_index], data_normalized_centered[test_index]
            t_train, t_test = labels[train_index].T[0], labels[test_index].T[0]

        if classifier == 0 or classifier == 1:
            ridge_classifier = MyRidgeRegression(x_train, t_train, x_test, t_test)
            best_lamb = ridge_classifier.grid_search()
            print("Grid Search final hyper-parameters :", best_lamb)
            ridge_classifier = MyRidgeRegression(x_train, t_train, x_test, t_test, lamb=best_lamb)
            ridge_classifier.training()
            print("Train accuracy : {:.4%}".format(
                ridge_classifier.classifier.score(ridge_classifier.x_train, ridge_classifier.t_train)))
            ridge_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(ridge_classifier.t_test, ridge_classifier.train_predictions)))

        if classifier == 0 or classifier == 2:
            discriminant_analysis_classifier = MyDiscriminantAnalysis(x_train, t_train, x_test, t_test)
            best_shrinkage = discriminant_analysis_classifier.grid_search()
            print("Grid Search final hyper-parameters :", best_shrinkage)
            discriminant_analysis_classifier = MyDiscriminantAnalysis(x_train, t_train, x_test, t_test,
                                                                             shrinkage=best_shrinkage)
            discriminant_analysis_classifier.training()
            print("Train accuracy : {:.4%}".format(
                discriminant_analysis_classifier.classifier.score(
                    discriminant_analysis_classifier.x_train, discriminant_analysis_classifier.t_train)))
            discriminant_analysis_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(discriminant_analysis_classifier.t_test,
                               discriminant_analysis_classifier.train_predictions)))

        if classifier == 0 or classifier == 3:
            logistic_regression_classifier = MyLogisticRegression(x_train, t_train, x_test, t_test)
            best_c = logistic_regression_classifier.grid_search()
            print("Grid Search final hyper-parameters :", best_c)
            logistic_regression_classifier = MyLogisticRegression(x_train, t_train, x_test, t_test, c=best_c)
            logistic_regression_classifier.training()
            print("Train accuracy : {:.4%}".format(
                logistic_regression_classifier.classifier.score(logistic_regression_classifier.x_train,
                                                                logistic_regression_classifier.t_train)))
            logistic_regression_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(logistic_regression_classifier.t_test, logistic_regression_classifier.
                               train_predictions)))

        if classifier == 0 or classifier == 4:
            neural_network_classifier = MyNeuralNetwork(x_train, t_train, x_test, t_test)
            best_lamb, best_hidden_layer_sizes = neural_network_classifier.grid_search()
            print("Grid Search final hyper-parameters :", best_lamb, ", ", best_hidden_layer_sizes)
            neural_network_classifier = MyNeuralNetwork(x_train, t_train, x_test, t_test, lamb=best_lamb,
                                                        hidden_layer_sizes=best_hidden_layer_sizes)
            neural_network_classifier.training()
            print("Train accuracy : {:.4%}".format(
                neural_network_classifier.classifier.score(neural_network_classifier.x_train,
                                                           neural_network_classifier.t_train)))
            neural_network_classifier.prediction()
            print("Test accuracy: {:.4%}".format(accuracy_score(neural_network_classifier.t_test,
                                                                neural_network_classifier.train_predictions)))

        if classifier == 0 or classifier == 5:
            perceptron_classifier = MyPerceptron(x_train, t_train, x_test, t_test)
            best_lamb, best_eta0 = perceptron_classifier.grid_search()
            print("Grid Search final hyper-parameters :", best_lamb, ", ", best_eta0)
            perceptron_classifier = MyPerceptron(x_train, t_train, x_test, t_test, lamb=best_lamb, eta0=best_eta0)
            perceptron_classifier.training()
            print("Train accuracy : {:.4%}".format(
                perceptron_classifier.classifier.score(perceptron_classifier.x_train, perceptron_classifier.t_train)))
            perceptron_classifier.prediction()
            print("Test accuracy: {:.4%}".format(accuracy_score(perceptron_classifier.t_test,
                                                                perceptron_classifier.train_predictions)))

        if classifier == 0 or classifier == 6:
            svm_classifier = MySVM(x_train, t_train, x_test, t_test)
            best_c, best_gamma = svm_classifier.grid_search()
            print("Grid Search final hyper-parameters :", best_c, ", ", best_gamma)
            svm_classifier = MySVM(x_train, t_train, x_test, t_test, c=best_c, gamma=best_gamma)
            svm_classifier.training()
            print("Train accuracy : {:.4%}".format(svm_classifier.classifier.score(svm_classifier.x_train,
                                                                                   svm_classifier.t_train)))
            svm_classifier.prediction()
            print("Test accuracy: {:.4%}".format(accuracy_score(svm_classifier.t_test,
                                                                svm_classifier.train_predictions)))

    return


if __name__ == '__main__':
    main()
