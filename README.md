IFT712_Projet_de_session : Comparaison de 6 méthodes de classification de données
==============================

Ce projet de session a pour but de comparer 6 classifieurs différents : AdaBoost, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), Logistic Regression, Neural Networks, Perceptron, Ridge Regression, Support Vector Machines et Naive Bayes. Les points de comparaison principaux sont le type de données, le type de traitement des données requis, le temps d'exécution, la log loss et le score de test et d’entraînement. Nous passons également en revue quelques bonnes pratiques à appliquer et comparons nos résultats à d’autres personnes ayant utilisé le même ensemble de données provenant du challenge Kaggle “Leaf Classification”.

***

## Prérequis
Pour fonctionner, ce projet a besoin d'un certain nombre de bibliothèques python que vous trouverez dans requirements.txt

Pour les installer, lancez ```pip install -r requirements.txt```

***

## Fonctionnement
Il est possible de lancer ce programme à partir de la commande suivante :

```python
python main.py train_data_input_filepath output_filepath classifier grid_search data_preprocessing use_pca

    classifier : 0=>All, 1=>Neural Networks, 2=>Linear_Discriminant_Analysis, 3=>Logistic_Regression, 4=Ridge, 5=>Perceptron, 6=>SVM, 7=> AdaBoost, 8=>Quadratic_Discriminant Analysis, 9=>Naive_Bayes, 10=Class_grouping

    grid_search : 0=>no grid search, 1=>use grid search

    data_preprocessing : 0=>raw data, 1=>centered + standard deviation normalization, 2=>centered + mean deviation normalization

    use_pca : 0=>no, 1=>yes
```
Exemple (Windows): ```python main.py data\\raw\\train\\leaf-classification-train.csv data\\processed 0 0 0 0```

Exemple (Linux): ```python main.py data/raw/train/leaf-classification-train.csv data/processed 0 0 0 0```

***

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
