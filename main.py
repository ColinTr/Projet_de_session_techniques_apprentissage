import sys

from src.data.data_handler import DataHandler
from src.data.data_handler import parse_csv_file
from sklearn.model_selection import StratifiedShuffleSplit
from src.models.cross_validation_utilities import sampling


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

        """print(data)
        print(data_normalized_centered)
        print(labels)
        print(species)"""

        #Create a set of data_train and data_test
        sss = StratifiedShuffleSplit(10, test_size=0.2)
        train_index, test_index = sss.split(data, labels)

        print(train_index)
        print(" \n ============= \n ")
        print(test_index)
        x_train, x_test = data[train_index], data[test_index]
        t_train, t_test = labels[train_index], labels[test_index]

        #(x_train, t_train)


        print("Done")
    return


if __name__ == '__main__':
    main()
