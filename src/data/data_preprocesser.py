from sklearn.model_selection import StratifiedShuffleSplit
from src.data.data_handler import DataHandler
from scipy.stats import normaltest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np


def apply_pca_on_data(data):
    pca = PCA(n_components='mle', svd_solver='full')
    return pca.fit_transform(data)


class DataPreprocesser:
    def __init__(self, data_input_filepath, output_filepath, data_preprocessing_method, use_pca):
        self.data_input_filepath = data_input_filepath
        self.output_filepath = output_filepath
        self.data_preprocessing_method = data_preprocessing_method
        self.use_pca = use_pca

        self.raw_data = self.data_normalized_centered = self.labels = self.species = None

    def read_data(self):
        print("=============== Reading and handling data ===============")
        dh = DataHandler(self.data_input_filepath, self.output_filepath)
        dh.main()
        self.raw_data, self.data_normalized_centered, self.labels, self.species = dh.read_all_output_files()
        return self.raw_data, self.data_normalized_centered, self.labels, self.species

    def apply_preprocessing(self):
        if self.data_preprocessing_method == 2:
            scaler = MinMaxScaler()
            scaler.fit(self.raw_data)
            self.data_normalized_centered = scaler.transform(self.raw_data)

        if self.use_pca == 1:
            data_descriptors_before = self.raw_data.shape[1]
            self.raw_data = apply_pca_on_data(self.raw_data)
            self.data_normalized_centered = apply_pca_on_data(self.data_normalized_centered)
            if self.data_preprocessing_method == 0:
                print("raw_data : Number of dimensions before PCA: " +
                      '{:1.0f}'.format(data_descriptors_before) + " after PCA: " +
                      '{:1.0f}'.format(self.raw_data.shape[1]))
            if self.data_preprocessing_method == 1 or self.data_preprocessing_method == 2:
                print("data_normalized_centered : Number of dimensions before PCA: " +
                      '{:1.0f}'.format(data_descriptors_before) + " after PCA: " +
                      '{:1.0f}'.format(self.data_normalized_centered.shape[1]))

        # We check that our data was correctly centered and normalized
        print("Mean of centered and normalized data :{:.4}".format(self.data_normalized_centered.mean()))
        print("Standard deviation of centered and normalized data :{:.4}".format(self.data_normalized_centered.std()))

        # ============================= TESTING FOR NORMALITY =============================
        normal_test = normaltest(self.raw_data)
        stat, p = normal_test
        print("Normaltest mean p={:.6}".format(np.mean(p, axis=0)))

        # ============================== GENERATING DATASETS ==============================
        # Let's create a train and test dataset
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
        # We take the first split of our sss
        x_train = x_test = t_train = t_test = None
        if self.data_preprocessing_method == 0:
            train_index, test_index = next(sss.split(self.raw_data, self.labels))
            x_train, x_test = self.raw_data[train_index], self.raw_data[test_index]
            t_train, t_test = self.labels[train_index].T[0], self.labels[test_index].T[0]
        if self.data_preprocessing_method == 1 or self.data_preprocessing_method == 2:
            train_index, test_index = next(sss.split(self.data_normalized_centered, self.labels))
            x_train, x_test = self.data_normalized_centered[train_index], self.data_normalized_centered[test_index]
            t_train, t_test = self.labels[train_index].T[0], self.labels[test_index].T[0]

        return x_train, t_train, x_test, t_test, self.species
