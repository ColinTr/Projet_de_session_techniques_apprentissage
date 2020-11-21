IFT712_Projet_de_session
==============================

Projet de session de techniques d'apprentissage : Comparaison de 6 méthodes de classification de données


Il est possible de lancer ce programme à partir de la commande suivante :

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python main.py train_data_input_filepath output_filepath classifier grid_search data_preprocessing use_pca

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;classifier : 0=>All, 1=>Neural Networks, 2=>Linear Discriminant Analysis, 3=>Logistic Regression, 4=Ridge, 5=>Perceptron, 6=>SVM, 7=> AdaBoost, 8=>Quadratic Discriminant Analysis, 9=>Naive Bayes, 10=Class grouping

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grid_search : 0=>no grid search, 1=>use grid search

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data_preprocessing : 0=>raw data, 1=>centered + standard deviation normalization, 2=>centered + mean deviation normalization

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;use_pca : 0=>no, 1=>yes

Exemple (Windows): python main.py data\\raw\\train\\leaf-classification-train.csv data\\processed 0 0 0 0

Exemple (Linux): python main.py data/raw/train/leaf-classification-train.csv data/processed 0 0 0 0

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │       ├── train      <- Data used to train the model.
    │       └── test       <- Data used to test the model.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── data_handler.py
    │   │   └── data_preprocesser.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   ├── adaboost_classifier.py
    │   │   ├── base_classifier.py
    │   │   ├── linear_discriminant_analysis.py
    │   │   ├── logistic_regression.py
    │   │   ├── naive_bayes.py
    │   │   ├── neural_networks.py
    │   │   ├── perceptron.py
    │   │   ├── quadratic_discriminant_analysis.py
    │   │   ├── ridge_regression.py
    │   │   ├── super_classifier.py
    │   │   └── support_vector_machines.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   │
    │   └── main.py        <- main script that launches everything needed to generate the results
    │
    └── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
                              generated with `pip freeze > requirements.txt`


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
