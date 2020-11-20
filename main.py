import sys

from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

from src.data.data_preprocesser import DataPreprocesser

from src.models.quadratic_discriminant_analysis import MyQuadraticDiscriminantAnalysis
from src.models.linear_discriminant_analysis import MyLinearDiscriminantAnalysis
from src.models.logistic_regression import MyLogisticRegression
from src.models.adaboost_classifier import MyAdaboostClassifier
from src.models.ridge_regression import MyRidgeRegression
from src.models.super_classifier import SuperClassifier
from src.models.neural_networks import MyNeuralNetwork
from src.models.support_vector_machines import MySVM
from src.models.naive_bayes import MyNaiveBayes
from src.models.perceptron import MyPerceptron


def main():
    if len(sys.argv) < 7:
        print("Usage: python main.py train_data_input_filepath output_filepath classifier grid_search"
              "data_preprocessing use_pca\n")
        print("classifier : 0=>All, 1=>Neural Networks, 2=>Linear Discriminant Analysis, 3=>Logistic Regression,"
              " 4=Ridge, 5=>Perceptron, 6=>SVM, 7=> AdaBoost, 8=>Quadratic Discriminant Analysis, 9=>Naive Bayes,"
              " 10=Class grouping\n")
        print("grid_search : 0=>no grid search, 1=>use grid search\n")
        print("data_preprocessing : 0=>raw data, 1=>centered+standard deviation normalization,"
              "2=>centered+min/max normalization\n")
        print("use_pca : 0=>no, 1=>yes\n")
        print("Exemple (Windows): python main.py data\\raw\\train\\leaf-classification-train.csv data\\processed 0 0 "
              "0 0\n")
        print("Exemple (Linux): python main.py data/raw/train/leaf-classification-train.csv data/processed 0 0 0 0\n")

    else:
        data_input_filepath = sys.argv[1]
        output_filepath = sys.argv[2]
        classifier = int(sys.argv[3])
        grid_search = int(sys.argv[4])
        data_preprocessing_method = int(sys.argv[5])
        use_pca = int(sys.argv[6])

        if use_pca != 0 and use_pca != 1:
            print("Incorrect value for parameter use_pca")
            return

        if classifier < 0 or classifier > 10:
            print("Incorrect value for parameter classifier")
            return

        if grid_search != 0 and grid_search != 1:
            print("Incorrect value for parameter grid_search")
            return

        if data_preprocessing_method != 0 and data_preprocessing_method != 1 and data_preprocessing_method != 2:
            print("Incorrect value for parameter centered_normalized_bool")
            return

        data_preprocesser = DataPreprocesser(data_input_filepath, output_filepath, data_preprocessing_method, use_pca)

        raw_data, data_normalized_centered, labels, species = data_preprocesser.read_data()

        x_train, t_train, x_test, t_test, species = data_preprocesser.apply_preprocessing()

        # ================================= NN GRID SEARCH =================================
        if classifier == 0 or classifier == 1:
            print("NEURAL NETWORK :")
            neural_network_classifier = MyNeuralNetwork(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                # best_lamb, best_hidden_layer_sizes = neural_network_classifier.grid_search()
                best_lamb, best_hidden_layer_sizes = neural_network_classifier.sklearn_random_grid_search(20)
                neural_network_classifier = MyNeuralNetwork(x_train, t_train, x_test, t_test, lamb=best_lamb,
                                                            hidden_layer_sizes=best_hidden_layer_sizes)

            print_results(neural_network_classifier, x_test, t_test)

        # ===================== LINEAR DISCRIMINANT ANALYSIS GRID SEARCH ===================
        if classifier == 0 or classifier == 2:
            print("LINEAR DISCRIMINANT :")
            discriminant_analysis_classifier = MyLinearDiscriminantAnalysis(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                # best_shrinkage = discriminant_analysis_classifier.grid_search()
                best_shrinkage = discriminant_analysis_classifier.sklearn_random_grid_search(50)
                discriminant_analysis_classifier = MyLinearDiscriminantAnalysis(x_train, t_train, x_test, t_test,
                                                                                shrinkage=best_shrinkage)

            print_results(discriminant_analysis_classifier, x_test, t_test)

        # ========================= LOGISTIC REGRESSION GRID SEARCH ========================
        if classifier == 0 or classifier == 3:
            print("LOGISTIC REGRESSION :")
            logistic_regression_classifier = MyLogisticRegression(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                # best_c = logistic_regression_classifier.grid_search()
                best_c = logistic_regression_classifier.sklearn_random_grid_search(30)
                logistic_regression_classifier = MyLogisticRegression(x_train, t_train, x_test, t_test, c=best_c)

            print_results(logistic_regression_classifier, x_test, t_test)

        # =========================== RIDGE REGRESSION GRID SEARCH =========================
        if classifier == 0 or classifier == 4:
            print("RIDGE REGRESSION :")
            ridge_classifier = MyRidgeRegression(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                # best_lamb = ridge_classifier.grid_search()
                best_lamb = ridge_classifier.sklearn_random_grid_search(30)
                ridge_classifier = MyRidgeRegression(x_train, t_train, x_test, t_test, lamb=best_lamb)

            ridge_classifier.training()
            print("Train accuracy : {:.4%}".format(ridge_classifier.classifier.score(
                ridge_classifier.x_train, ridge_classifier.t_train)))

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
            print("Train accuracy : {:.4%}".format(perceptron_classifier.classifier.score(
                perceptron_classifier.x_train, perceptron_classifier.t_train)))

            perceptron_classifier.prediction()
            print("Test accuracy : {:.4%}".format(
                accuracy_score(perceptron_classifier.t_test, perceptron_classifier.train_predictions)))

        # ================================= SVM GRID SEARCH ================================
        if classifier == 0 or classifier == 6:
            print("SVM :")
            svm_classifier = MySVM(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                # best_c, best_gamma = svm_classifier.grid_search()
                best_c, best_gamma = svm_classifier.sklearn_random_grid_search(50)
                svm_classifier = MySVM(x_train, t_train, x_test, t_test, c=best_c, gamma=best_gamma)

            print_results(svm_classifier, x_test, t_test)

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

            print_results(adaboost_classifier, x_test, t_test)

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

            print_results(quadratic_discriminant_analysis_classifier, x_test, t_test)

        # ============================= NAIVE BAYES GRID SEARCH ============================
        if classifier == 0 or classifier == 9:
            print("NAIVE BAYES :")
            gaussian_naive_bayes = MyNaiveBayes(x_train, t_train, x_test, t_test)
            if grid_search == 1:
                # best_smoothing = gaussian_naive_bayes.grid_search()
                best_smoothing = gaussian_naive_bayes.sklearn_random_grid_search(50)
                gaussian_naive_bayes = MyNaiveBayes(x_train, t_train, x_test, t_test, var_smoothing=best_smoothing)

            print_results(gaussian_naive_bayes, x_test, t_test)

        # ================================= CLASS GROUPING =================================
        if classifier == 0 or classifier == 10:
            print("CLASS GROUPING :")
            super_classifier_data = raw_data
            if data_preprocessing_method == 1:
                super_classifier_data = data_normalized_centered
            if use_pca == 1:
                super_classifier_data = PCA(n_components='mle', svd_solver='full').fit_transform(super_classifier_data)
            super_classifier = SuperClassifier(super_classifier_data, species, None, grid_search=grid_search)
            super_classifier.train_base_classifier()
            super_classifier.calculate_test_and_train_accuracy()

    return


def print_results(classifier, x_test, t_test):
    classifier.training()
    print("Train accuracy : {:.4%}".format(classifier.classifier.score(
        classifier.x_train, classifier.t_train)))
    print("Train log loss = ", log_loss(classifier.t_train, classifier.classifier.predict_proba(classifier.x_train)))

    classifier.prediction()

    print("Test accuracy : {:.4%}".format(
        accuracy_score(classifier.t_test, classifier.train_predictions)))
    print("Test log loss = ", log_loss(t_test, classifier.classifier.predict_proba(x_test)))
    return


if __name__ == '__main__':
    main()
