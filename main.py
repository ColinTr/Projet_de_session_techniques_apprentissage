import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

from src.data.data_handler import DataHandler
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
from sklearn.preprocessing import MinMaxScaler


def apply_pca_on_data(data):
    pca = PCA(n_components='mle', svd_solver='full')
    return pca.fit_transform(data)


def main():
    if len(sys.argv) < 7:
        print("Usage: python main.py train_data_input_filepath output_filepath classifier grid_search"
              "data_preprocessing use_pca\n")
        print("classifier : 0=>All, 1=>Neural Networks, 2=>Linear Discriminant Analysis, 3=>Logistic Regression,"
              " 4=Ridge, 5=>Perceptron, 6=>SVM, 7=> AdaBoost, 8=>Quadratic Discriminant Analysis, 9=>Naive Bayes\n")
        print("grid_search : 0=>no grid search, 1=>use grid search\n")
        print("data_preprocessing : 0=>raw data, 1=>centered+standard deviation normalization,"
              "2=>centered+min/max normalization\n")
        print("use_pca : 0=>no, 1=>yes\n")
        print("Exemple (Windows): python main.py data\\raw\\train\\leaf-classification-train.csv data\\processed 0 1 "
              "1 0\n")
        print("Exemple (Linux): python main.py data/raw/train/leaf-classification-train.csv data/processed 0 1 1 0\n")

    else:
        data_input_filepath = sys.argv[1]
        output_filepath = sys.argv[2]
        classifier = int(sys.argv[3])
        grid_search = int(sys.argv[4])
        data_preprocessing = int(sys.argv[5])
        use_pca = int(sys.argv[6])

        if use_pca != 0 and use_pca != 1:
            print("Incorrect value for parameter use_pca")
            return

        if classifier < 0 or classifier > 9:
            print("Incorrect value for parameter classifier")
            return

        if grid_search != 0 and grid_search != 1:
            print("Incorrect value for parameter grid_search")
            return

        if data_preprocessing != 0 and data_preprocessing != 1 and data_preprocessing != 2:
            print("Incorrect value for parameter centered_normalized_bool")
            return

        print("=============== Reading and handling data ===============")
        dh = DataHandler(data_input_filepath, output_filepath)
        dh.main()
        raw_data, data_normalized_centered, labels, species = \
            dh.read_all_output_files()

        if data_preprocessing == 2:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data_normalized_centered)
            data_normalized_centered = scaler.transform(data_normalized_centered)

        if use_pca == 1:
            data_descriptors_before = raw_data.shape[1]
            raw_data = apply_pca_on_data(raw_data)
            data_normalized_centered = apply_pca_on_data(data_normalized_centered)
            if data_preprocessing == 0:
                print("raw_data : Number of dimensions before PCA: " +
                      '{:1.0f}'.format(data_descriptors_before) + " after PCA: " +
                      '{:1.0f}'.format(raw_data.shape[1]))
            if data_preprocessing == 1 or data_preprocessing == 2:
                print("data_normalized_centered : Number of dimensions before PCA: " +
                      '{:1.0f}'.format(data_descriptors_before) + " after PCA: " +
                      '{:1.0f}'.format(data_normalized_centered.shape[1]))

        # We check that our data was correctly centerd and normalized
        print("Mean of centered and normalized data :{:.4}".format(data_normalized_centered.mean()))
        print("Standard deviation of centered and normalized data :{:.4}".format(data_normalized_centered.std()))

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
        if data_preprocessing == 0:
            train_index, test_index = next(sss.split(raw_data, labels))
            x_train, x_test = raw_data[train_index], raw_data[test_index]
            t_train, t_test = labels[train_index].T[0], labels[test_index].T[0]
        if data_preprocessing == 1 or data_preprocessing == 2:
            train_index, test_index = next(sss.split(data_normalized_centered, labels))
            x_train, x_test = data_normalized_centered[train_index], data_normalized_centered[test_index]
            t_train, t_test = labels[train_index].T[0], labels[test_index].T[0]

        # ================================= NN GRID SEARCH =================================
        if classifier == 0 or classifier == 1:
            print("NEURAL NETWORK :")
            neural_network_classifier = MyNeuralNetwork(x_train, t_train, x_test, t_test)
            if grid_search == 1:
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
            print("LINEAR DISCRIMINANT :")
            discriminant_analysis_classifier = MyLinearDiscriminantAnalysis(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                # best_shrinkage = discriminant_analysis_classifier.grid_search()
                best_shrinkage = discriminant_analysis_classifier.sklearn_random_grid_search(50)
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
            print("LOGISTIC REGRESSION :")
            logistic_regression_classifier = MyLogisticRegression(x_train, t_train, x_test, t_test)
            if grid_search == 1:
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
            print("RIDGE REGRESSION :")
            ridge_classifier = MyRidgeRegression(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                # best_lamb = ridge_classifier.grid_search()
                best_lamb = ridge_classifier.sklearn_random_grid_search(30)
                ridge_classifier = MyRidgeRegression(x_train, t_train, x_test, t_test, lamb=best_lamb)
            ridge_classifier.training()
            print("Train accuracy : {:.4%}".format(
                ridge_classifier.classifier.score(ridge_classifier.x_train, ridge_classifier.t_train)))
            ridge_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(ridge_classifier.t_test, ridge_classifier.train_predictions)))

        # ============================= PERCEPTRON GRID SEARCH =============================
        if classifier == 0 or classifier == 5:
            print("PERCEPTRON :")
            perceptron_classifier = MyPerceptron(x_train, t_train, x_test, t_test)
            if grid_search == 1:
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
            print("SVM :")
            svm_classifier = MySVM(x_train, t_train, x_test, t_test)
            if grid_search == 1:
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
            print("ADABOOST :")
            adaboost_classifier = MyAdaboostClassifier(x_train, t_train, x_test, t_test,
                                                       DecisionTreeClassifier(max_depth=1))
            if grid_search == 1:
                # best_learning_rate, best_n_estimators, best_base_estimator = adaboost_classifier.grid_search()
                best_learning_rate, best_n_estimators, best_base_estimator = adaboost_classifier.\
                    sklearn_random_grid_search(50)

                adaboost_classifier = MyAdaboostClassifier(x_train, t_train, x_test, t_test, best_base_estimator,
                                                           learning_rate=best_learning_rate,
                                                           n_estimators=best_n_estimators)
            adaboost_classifier.training()
            print("Train accuracy : {:.4%}".format(adaboost_classifier.classifier.score(adaboost_classifier.x_train,
                                                                                        adaboost_classifier.t_train)))
            adaboost_classifier.prediction()
            print("Test accuracy: {:.4%}".format(accuracy_score(adaboost_classifier.t_test,
                                                                adaboost_classifier.train_predictions)))

        # =================== QUADRATIC DISCRIMINANT ANALYSIS GRID SEARCH ==================
        if classifier == 0 or classifier == 8:
            print("QUADRATIC DISCRIMINANT :")
            quadratic_discriminant_analysis_classifier = MyQuadraticDiscriminantAnalysis(x_train, t_train,
                                                                                         x_test, t_test)
            if grid_search == 1:
                # best_shrinkage = discriminant_analysis_classifier.grid_search()
                best_reg_param, best_store_covariance = quadratic_discriminant_analysis_classifier. \
                    sklearn_random_grid_search(35)
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

        # ============================= NAIVE BAYES GRID SEARCH ============================
        if classifier == 0 or classifier == 9:
            print("NAIVE BAYES :")
            gaussian_naive_bayes = MyNaiveBayes(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                best_smoothing = gaussian_naive_bayes.sklearn_random_grid_search(50)
                gaussian_naive_bayes = MyNaiveBayes(x_train, t_train, x_test, t_test, var_smoothing=best_smoothing)

            gaussian_naive_bayes.training()
            print("Train accuracy : {:.4%}".format(gaussian_naive_bayes.classifier.score(
                gaussian_naive_bayes.x_train, gaussian_naive_bayes.t_train)))

            gaussian_naive_bayes.prediction()

            print("Test accuracy : {:.4%}".format(
                accuracy_score(gaussian_naive_bayes.t_test, gaussian_naive_bayes.train_predictions)))

    return


if __name__ == '__main__':
    main()
