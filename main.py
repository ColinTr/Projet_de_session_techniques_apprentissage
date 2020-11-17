import sys

from sklearn.metrics import accuracy_score
from src.data.data_handler import DataHandler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA

from src.models.adaboost_classifier import MyAdaboostClassifier
from src.models.linear_discriminant_analysis import MyLinearDiscriminantAnalysis
from src.models.neural_networks import MyNeuralNetwork
from src.models.perceptron import MyPerceptron
from src.models.logistic_regression import MyLogisticRegression
from src.models.quadratic_discriminant_analysis import MyQuadraticDiscriminantAnalysis
from src.models.support_vector_machines import MySVM
from src.models.ridge_regression import MyRidgeRegression
from src.models.naive_bayes import MyNaiveBayes

from scipy.stats import normaltest


def apply_pca_on_data(data):
    pca = PCA(n_components='mle', svd_solver='full')
    return pca.fit_transform(data)


def main():
    if len(sys.argv) < 6:
        print("Usage: python data_handler.py train_data_input_filepath output_filepath classifier "
              "centered_normalized_data use_pca\n")
        print("classifier : 0=>all, 1=>>neural networks, 2=>linear discriminant analysis, 3=>logistic,"
              " 4=ridge, 5=>perceptron, 6=>SVM, 7=> AdaBoost, 8=>quadratic discriminant analysis\n")
        print("centered_normalized_data : 0=>raw data, 1=>centered and normalized data\n")
        print("use_pca : 0=>no, 1=>yes\n")
        print("Exemple (Windows): python main.py data\\raw\\train\\leaf-classification-train.csv data\\processed 0 1\n")
        print("Exemple (Linux): python main.py data/raw/train/leaf-classification-train.csv data/processed 0 1\n")

    else:
        data_input_filepath = sys.argv[1]
        output_filepath = sys.argv[2]
        classifier = int(sys.argv[3])
        centered_normalized_bool = int(sys.argv[4])
        use_pca = int(sys.argv[5])

        if centered_normalized_bool != 0 and centered_normalized_bool != 1:
            print("Incorrect value in centered_normalized_data parameter")
            return

        if use_pca != 0 and use_pca != 1:
            print("Incorrect value in use_pca parameter")
            return

        if classifier < 0 or classifier > 9:
            print("Incorrect value in classifier parameter")
            return

        print("=============== Reading and handling data ===============")
        dh = DataHandler(data_input_filepath, output_filepath)
        dh.main()
        raw_data, data_normalized_centered, labels, species = \
            dh.read_all_output_files()

        if use_pca == 1:
            data_descriptors_before = raw_data.shape[1]
            raw_data = apply_pca_on_data(raw_data)
            data_normalized_centered = apply_pca_on_data(data_normalized_centered)
            print("raw_data : number of descriptors before PCA: " +
                  '{:1.0f}'.format(data_descriptors_before) + " after PCA: " +
                  '{:1.0f}'.format(raw_data.shape[1]))
            print("data_normalized_centered : number of descriptors before PCA: " +
                  '{:1.0f}'.format(data_descriptors_before) + " after PCA: " +
                  '{:1.0f}'.format(data_normalized_centered.shape[1]))

        # We check that our data preprocessing was correctly done
        print("Centered and normalized data mean :{:.4}".format(data_normalized_centered.mean()))
        print("Centered and normalized data standard deviation :{:.4}".format(data_normalized_centered.std()))

        # ============================= TESTING FOR NORMALITY =============================
        p_total = 0
        for i in range(0, len(data_normalized_centered[0])):
            column = []
            for j in range(0, len(data_normalized_centered)):
                column.append(data_normalized_centered[j, i])
            stat, p = normaltest(column)
            p_total += p
        print("Normaltest mean p={:.4}".format(p_total / len(data_normalized_centered)))

        # ============================== GENERATING DATASETS ==============================
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

        # ================================= NN GRID SEARCH =================================
        if classifier == 0 or classifier == 1:
            neural_network_classifier = MyNeuralNetwork(x_train, t_train, x_test, t_test)
            # best_lamb, best_hidden_layer_sizes = neural_network_classifier.grid_search()
            best_lamb, best_hidden_layer_sizes = neural_network_classifier.sklearn_random_grid_search(20)
            neural_network_classifier = MyNeuralNetwork(x_train, t_train, x_test, t_test, lamb=best_lamb,
                                                        hidden_layer_sizes=best_hidden_layer_sizes)
            neural_network_classifier.training()
            print("Train accuracy : {:.4%}".format(
                neural_network_classifier.classifier.score(neural_network_classifier.x_train,
                                                           neural_network_classifier.t_train)))
            neural_network_classifier.prediction()
            print("Test accuracy: {:.4%}".format(accuracy_score(neural_network_classifier.t_test,
                                                                neural_network_classifier.train_predictions)))

        # ===================== LINEAR DISCRIMINANT ANALYSIS GRID SEARCH ===================
        if classifier == 0 or classifier == 2:
            discriminant_analysis_classifier = MyLinearDiscriminantAnalysis(x_train, t_train, x_test, t_test)
            # best_shrinkage = discriminant_analysis_classifier.grid_search()
            best_shrinkage = discriminant_analysis_classifier.sklearn_random_grid_search(100)
            discriminant_analysis_classifier = MyLinearDiscriminantAnalysis(x_train, t_train, x_test, t_test,
                                                                            shrinkage=best_shrinkage)
            discriminant_analysis_classifier.training()
            print("Train accuracy : {:.4%}".format(
                discriminant_analysis_classifier.classifier.score(
                    discriminant_analysis_classifier.x_train, discriminant_analysis_classifier.t_train)))
            discriminant_analysis_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(discriminant_analysis_classifier.t_test,
                               discriminant_analysis_classifier.train_predictions)))

        # ========================= LOGISTIC REGRESSION GRID SEARCH ========================
        if classifier == 0 or classifier == 3:
            logistic_regression_classifier = MyLogisticRegression(x_train, t_train, x_test, t_test)
            # best_c = logistic_regression_classifier.grid_search()
            best_c = logistic_regression_classifier.sklearn_random_grid_search(30)
            logistic_regression_classifier = MyLogisticRegression(x_train, t_train, x_test, t_test, c=best_c)
            logistic_regression_classifier.training()
            print("Train accuracy : {:.4%}".format(
                logistic_regression_classifier.classifier.score(logistic_regression_classifier.x_train,
                                                                logistic_regression_classifier.t_train)))
            logistic_regression_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(logistic_regression_classifier.t_test, logistic_regression_classifier.
                               train_predictions)))

        # =========================== RIDGE REGRESSION GRID SEARCH =========================
        if classifier == 0 or classifier == 4:
            ridge_classifier = MyRidgeRegression(x_train, t_train, x_test, t_test)
            # best_lamb = ridge_classifier.grid_search()
            best_lamb = ridge_classifier.sklearn_random_grid_search(100)

            ridge_classifier = MyRidgeRegression(x_train, t_train, x_test, t_test, lamb=best_lamb)
            ridge_classifier.training()
            print("Train accuracy : {:.4%}".format(
                ridge_classifier.classifier.score(ridge_classifier.x_train, ridge_classifier.t_train)))
            ridge_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(ridge_classifier.t_test, ridge_classifier.train_predictions)))

        # ============================= PERCEPTRON GRID SEARCH =============================
        if classifier == 0 or classifier == 5:
            perceptron_classifier = MyPerceptron(x_train, t_train, x_test, t_test)
            # best_lamb, best_eta0 = perceptron_classifier.grid_search()
            best_lamb, best_eta0 = perceptron_classifier.sklearn_random_grid_search(50)
            perceptron_classifier = MyPerceptron(x_train, t_train, x_test, t_test, lamb=best_lamb, eta0=best_eta0)
            perceptron_classifier.training()
            print("Train accuracy : {:.4%}".format(
                perceptron_classifier.classifier.score(perceptron_classifier.x_train, perceptron_classifier.t_train)))
            perceptron_classifier.prediction()
            print("Test accuracy: {:.4%}".format(accuracy_score(perceptron_classifier.t_test,
                                                                perceptron_classifier.train_predictions)))

        # ================================= SVM GRID SEARCH ================================
        if classifier == 0 or classifier == 6:
            svm_classifier = MySVM(x_train, t_train, x_test, t_test)
            # best_c, best_gamma = svm_classifier.grid_search()
            best_c, best_gamma = svm_classifier.sklearn_random_grid_search(50)
            svm_classifier = MySVM(x_train, t_train, x_test, t_test, c=best_c, gamma=best_gamma)
            svm_classifier.training()
            print("Train accuracy : {:.4%}".format(svm_classifier.classifier.score(svm_classifier.x_train,
                                                                                   svm_classifier.t_train)))
            svm_classifier.prediction()
            print("Test accuracy: {:.4%}".format(accuracy_score(svm_classifier.t_test,
                                                                svm_classifier.train_predictions)))

        # ============================== ADABOOST GRID SEARCH ==============================
        if classifier == 0 or classifier == 7:
            adaboost_classifier = MyAdaboostClassifier(x_train, t_train, x_test, t_test, 'None')
            # best_learning_rate, best_n_estimators, best_base_estimator = adaboost_classifier.grid_search()
            best_learning_rate, best_n_estimators, best_base_estimator = adaboost_classifier.sklearn_random_grid_search(
                50)

            adaboost_classifier = MyAdaboostClassifier(x_train, t_train, x_test, t_test, best_base_estimator,
                                                       learning_rate=best_learning_rate, n_estimators=best_n_estimators)
            adaboost_classifier.training()
            print("Train accuracy : {:.4%}".format(adaboost_classifier.classifier.score(adaboost_classifier.x_train,
                                                                                        adaboost_classifier.t_train)))
            adaboost_classifier.prediction()
            print("Test accuracy: {:.4%}".format(accuracy_score(adaboost_classifier.t_test,
                                                                adaboost_classifier.train_predictions)))

        # =================== QUADRATIC DISCRIMINANT ANALYSIS GRID SEARCH ==================
        if classifier == 0 or classifier == 8:
            quadratic_discriminant_analysis_classifier = MyQuadraticDiscriminantAnalysis(x_train, t_train,
                                                                                         x_test, t_test)
            # best_shrinkage = discriminant_analysis_classifier.grid_search()
            best_reg_param, best_store_covariance = quadratic_discriminant_analysis_classifier. \
                sklearn_random_grid_search(100)
            quadratic_discriminant_analysis_classifier = \
                MyQuadraticDiscriminantAnalysis(x_train, t_train, x_test, t_test,
                                                reg_param=best_reg_param, store_covariance=best_store_covariance)
            quadratic_discriminant_analysis_classifier.training()
            print("Train accuracy : {:.4%}".format(
                quadratic_discriminant_analysis_classifier.classifier.score(
                    quadratic_discriminant_analysis_classifier.x_train,
                    quadratic_discriminant_analysis_classifier.t_train)))
            quadratic_discriminant_analysis_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(quadratic_discriminant_analysis_classifier.t_test,
                               quadratic_discriminant_analysis_classifier.train_predictions)))

        # ====================== Gaussian Naive Bayes =====================
        if classifier == 0 or classifier == 9:
            gaussian_naive_bayes = MyNaiveBayes(x_train, t_train, x_test, t_test)

            gaussian_naive_bayes.training()
            print("Train accuracy : {:.4%}".format(gaussian_naive_bayes.classifier.score(
                gaussian_naive_bayes.x_train, gaussian_naive_bayes.t_train)))

            gaussian_naive_bayes.prediction()

            print("Test accuracy : {:.4%}".format(
                accuracy_score(gaussian_naive_bayes.t_test, gaussian_naive_bayes.train_predictions)))

    return


if __name__ == '__main__':
    main()
