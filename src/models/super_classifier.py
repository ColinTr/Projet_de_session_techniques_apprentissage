from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np

from src.models.neural_networks import MyNeuralNetwork


class SuperClassifier:
    """
    Our implementation of the classification by sub classes.\n
    This algorithm trains a first algorithm that is capable of predicting
    the main specie of a specie (Acer_Opalus => Opalus).\n
    Then, it trains a sub classifier for every main specie that will be
    capable of recognizing the sub species of a main specie. \n
    \n
    So to predict a class from data, it will first predict the main class
    of the data using the main classifier, then it will use the corresponding
    sub classifier to predict the sub specie that will be our final result.
    """
    def __init__(self, data, data_species_column, base_classifier, grid_search=False):
        self.grid_search = grid_search
        self.data = data
        self.data_species_column = data_species_column

        self.data_super_species_names = None
        self.data_super_species_labels = None

        self.base_classifier = base_classifier
        self.sub_classifiers = None

        self.train_data_super_dict = None
        self.test_data_super_dict = None
        self.train_sub_labels_super_dict = None
        self.test_sub_labels_super_dict = None

        self.x_train = None
        self.t_train = None
        self.t_train_super_species_encoded_labels = None
        self.x_test = None
        self.t_test = None
        self.t_test_super_species_encoded_labels = None
        self.t_train_sub_species_encoded_labels = None
        self.t_test_sub_species_encoded_labels = None

    def train_base_classifier(self):
        # We create a list with the super classes names
        self.data_super_species_names = [specie[0].split("_")[0] for specie in self.data_species_column]

        # We encode this list
        le = LabelEncoder().fit(self.data_super_species_names)
        self.data_super_species_labels = le.transform(self.data_super_species_names)

        self.fill_dicts()

        self.base_classifier = MyNeuralNetwork(self.x_train, self.t_train_super_species_encoded_labels,
                                               self.x_test, self.t_test_super_species_encoded_labels)

        self.base_classifier.training()
        print("Super classifier train accuracy : {:.4%}".format(self.base_classifier.classifier.score(
            self.base_classifier.x_train, self.base_classifier.t_train)))
        self.base_classifier.prediction()
        print("Super classifier test accuracy : {:.4%}".format(
            accuracy_score(self.base_classifier.t_test, self.base_classifier.train_predictions)))

        if self.grid_search:
            best_lamb, best_hidden_layer_sizes = self.base_classifier.sklearn_random_grid_search(20)
            self.base_classifier = MyNeuralNetwork(self.x_train, self.t_train_super_species_encoded_labels,
                                                   self.x_test, self.t_test_super_species_encoded_labels,
                                                   lamb=best_lamb, hidden_layer_sizes=best_hidden_layer_sizes)
        self.base_classifier.training()
        self.base_classifier.prediction()
        print("Super classifier test accuracy after grid search : {:.4%}".format(
            accuracy_score(self.base_classifier.t_test, self.base_classifier.train_predictions)))

        self.sub_classifiers = dict()

        for key_name in self.train_data_super_dict:
            if len(Counter(self.train_sub_labels_super_dict[key_name]).values()) < 2:
                self.sub_classifiers[key_name] = 'NoSubClassifierRequired_' +\
                                                 str(self.train_sub_labels_super_dict[key_name][0])
            else:
                sub_x_train = self.train_data_super_dict[key_name]
                sub_t_train = self.train_sub_labels_super_dict[key_name]
                sub_x_test = self.test_data_super_dict[key_name]
                sub_t_test = self.test_sub_labels_super_dict[key_name]
                self.sub_classifiers[key_name] = MyNeuralNetwork(sub_x_train, sub_t_train, sub_x_test, sub_t_test)
                if self.grid_search:
                    best_lamb, best_hidden_layer_sizes = self.sub_classifiers[key_name].sklearn_random_grid_search(10)
                    self.sub_classifiers[key_name] = MyNeuralNetwork(sub_x_train, sub_t_train, sub_x_test, sub_t_test,
                                                                     hidden_layer_sizes=best_hidden_layer_sizes,
                                                                     lamb=best_lamb)
                self.sub_classifiers[key_name].training()
                print("Sub classifier train accuracy : {:.4%}".format(self.sub_classifiers[key_name].classifier.score(
                    self.sub_classifiers[key_name].x_train, self.sub_classifiers[key_name].t_train)))
                self.sub_classifiers[key_name].prediction()
                print("Sub classifier test accuracy : {:.4%}".format(
                    accuracy_score(self.sub_classifiers[key_name].t_test,
                                   self.sub_classifiers[key_name].train_predictions)))

        return

    def calculate_test_and_train_accuracy(self):
        print("Final algorithm train accuracy : {:.4%}".format(
            accuracy_score(self.execute_class_grouping_prediction_algorithm(self.x_train),
                           self.t_train_sub_species_encoded_labels)))

        print("Final algorithm test accuracy : {:.4%}".format(
            accuracy_score(self.execute_class_grouping_prediction_algorithm(self.x_test),
                           self.t_test_sub_species_encoded_labels)))

        return

    def execute_class_grouping_prediction_algorithm(self, x_to_predict):
        final_predictions_list = []

        for x in x_to_predict:
            super_class_prediction = self.base_classifier.classifier.predict([x])[0]
            # If we find a string in the classifier list, it means this super class only has one class,
            # so we extract the sub class number
            if isinstance(self.sub_classifiers[super_class_prediction], str):
                final_predictions_list.append(np.int64(self.sub_classifiers[super_class_prediction].split("_")[1]))
            else:
                final_predictions_list.append(self.sub_classifiers[super_class_prediction]
                                              .classifier.predict([x])[0])

        return final_predictions_list

    def fill_dicts(self):
        sub_label_encoder = LabelEncoder().fit([x[0] for x in self.data_species_column])
        super_label_encoder = LabelEncoder().fit(self.data_super_species_names)

        # Dictionary with ENCODED super species names as keys and list of sub species as values
        self.train_data_super_dict = dict()
        self.test_data_super_dict = dict()

        # Same size dictionary but with the encoded sup species labels instead of the data
        self.train_sub_labels_super_dict = dict()
        self.test_sub_labels_super_dict = dict()

        # Let's create a train and test dataset for the super classifier
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
        train_index, test_index = next(sss.split(self.data, self.data_super_species_labels))

        self.x_train = self.data[train_index]
        self.x_test = self.data[test_index]
        self.t_train = [a[0] for a in self.data_species_column[train_index]]
        self.t_train_super_species_encoded_labels = [super_label_encoder.transform([a[0].split("_")[0]])[0]
                                                     for a in self.data_species_column[train_index]]
        self.t_test = [a[0] for a in self.data_species_column[test_index]]
        self.t_test_super_species_encoded_labels = [super_label_encoder.transform([a[0].split("_")[0]])[0]
                                                    for a in self.data_species_column[test_index]]
        self.t_train_sub_species_encoded_labels = [sub_label_encoder.transform([t])[0] for t in self.t_train]
        self.t_test_sub_species_encoded_labels = [sub_label_encoder.transform([t])[0] for t in self.t_test]

        for x, t in zip(self.x_train, self.t_train):
            super_specie_name = t.split("_")[0]
            encoded_super_specie_name = super_label_encoder.transform([super_specie_name])[0]
            if encoded_super_specie_name in self.train_data_super_dict.keys():
                self.train_data_super_dict[encoded_super_specie_name].append(x)
                self.train_sub_labels_super_dict[encoded_super_specie_name].append(sub_label_encoder.transform([t])[0])
            else:
                self.train_data_super_dict[encoded_super_specie_name] = [x]
                self.train_sub_labels_super_dict[encoded_super_specie_name] = [sub_label_encoder.transform([t])[0]]

        for x, t in zip(self.x_test, self.t_test):
            super_specie_name = t.split("_")[0]
            encoded_super_specie_name = super_label_encoder.transform([super_specie_name])[0]
            if encoded_super_specie_name in self.test_data_super_dict.keys():
                self.test_data_super_dict[encoded_super_specie_name].append(x)
                self.test_sub_labels_super_dict[encoded_super_specie_name].append(sub_label_encoder.transform([t])[0])
            else:
                self.test_data_super_dict[encoded_super_specie_name] = [x]
                self.test_sub_labels_super_dict[encoded_super_specie_name] = [sub_label_encoder.transform([t])[0]]

        return
