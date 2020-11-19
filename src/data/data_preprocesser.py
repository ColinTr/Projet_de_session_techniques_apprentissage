from sklearn.model_selection import StratifiedShuffleSplit
from src.data.data_handler import DataHandler
from scipy.stats import normaltest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def apply_pca_on_data(data):
    pca = PCA(n_components='mle', svd_solver='full')
    return pca.fit_transform(data)


class DataPreprocesser:
    def __init__(self, data_input_filepath, output_filepath, classifier, data_preprocessing_method, use_pca):
        self.data_input_filepath = data_input_filepath
        self.output_filepath = output_filepath
        self.classifier = classifier
        self.data_preprocessing_method = data_preprocessing_method
        self.use_pca = use_pca

    def apply_preprocessing(self):
        print("=============== Reading and handling data ===============")
        dh = DataHandler(self.data_input_filepath, self.output_filepath)
        dh.main()
        raw_data, data_normalized_centered, labels, species = \
            dh.read_all_output_files()

        if self.data_preprocessing_method == 2:
            scaler = MinMaxScaler()
            scaler.fit(raw_data)
            data_normalized_centered = scaler.transform(raw_data)

        if self.use_pca == 1:
            data_descriptors_before = raw_data.shape[1]
            raw_data = apply_pca_on_data(raw_data)
            data_normalized_centered = apply_pca_on_data(data_normalized_centered)
            if self.data_preprocessing_method == 0:
                print("raw_data : Number of dimensions before PCA: " +
                      '{:1.0f}'.format(data_descriptors_before) + " after PCA: " +
                      '{:1.0f}'.format(raw_data.shape[1]))
            if self.data_preprocessing_method == 1 or self.data_preprocessing_method == 2:
                print("data_normalized_centered : Number of dimensions before PCA: " +
                      '{:1.0f}'.format(data_descriptors_before) + " after PCA: " +
                      '{:1.0f}'.format(data_normalized_centered.shape[1]))

        # We check that our data was correctly centered and normalized
        print("Mean of centered and normalized data :{:.4}".format(data_normalized_centered.mean()))
        print("Standard deviation of centered and normalized data :{:.4}".format(data_normalized_centered.std()))

        # ============================= TESTING FOR NORMALITY =============================
        p_total = 0
        for i in range(0, len(raw_data[0])):
            column = []
            for j in range(0, len(raw_data)):
                column.append(raw_data[j, i])
            stat, p = normaltest(column)
            p_total += p
        print("Normaltest mean p={:.4}".format(p_total / len(data_normalized_centered)))

        # ============================== GENERATING DATASETS ==============================
        # Let's create a train and test dataset
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
        # We take the first split of our sss
        x_train = x_test = t_train = t_test = None
        if self.data_preprocessing_method == 0:
            train_index, test_index = next(sss.split(raw_data, labels))
            x_train, x_test = raw_data[train_index], raw_data[test_index]
            t_train, t_test = labels[train_index].T[0], labels[test_index].T[0]
        if self.data_preprocessing_method == 1 or self.data_preprocessing_method == 2:
            train_index, test_index = next(sss.split(data_normalized_centered, labels))
            x_train, x_test = data_normalized_centered[train_index], data_normalized_centered[test_index]
            t_train, t_test = labels[train_index].T[0], labels[test_index].T[0]

        return x_train, t_train, x_test, t_test
