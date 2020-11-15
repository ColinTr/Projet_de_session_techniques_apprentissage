import sys

from src.data.data_handler import DataHandler
from src.data.data_handler import parse_csv_file


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python data_handler.py test_data_input_filepath train_data_input_filepath output_filepath\n")
        print(
            "Exemple (Windows) : python main.py data\\raw\\train\\leaf-classification-train.csv "
            "data\\raw\\test\\leaf-classification-test.csv data\\processed\n")
        print("Exemple (Linux) : python main.py data/raw/train/leaf-classification-train.csv "
              "data/raw/test/leaf-classification-test.csv data/processed\n")

    else:
        train_data_input_filepath = sys.argv[1]
        test_data_input_filepath = sys.argv[2]
        output_filepath = sys.argv[3]

        print("=============== Reading and handling data ===============")
        dh = DataHandler(train_data_input_filepath, test_data_input_filepath, output_filepath)
        dh.main()
        train_data, train_data_normalized_centered, test_data, train_labels, train_species, test_ids =\
            dh.read_all_output_files()

        print(train_data)
        print(train_data_normalized_centered)
        print(test_data)
        print(train_labels)
        print(train_species)
        print(test_ids)

        print("Done")
    return


if __name__ == '__main__':
    main()
