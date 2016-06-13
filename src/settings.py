import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

CLASSIFIERS_SETTINGS = [
    {
        "classifier": GaussianNB,
        "name": "Gaussian Navie Bayes",
        "dir": "gaussian_nb",
        "args": {}
    },
    {
        "classifier": KNeighborsClassifier,
        "name": "Gaussian Navie Bayes",
        "dir": "knn",
        "args": {}
    },
    {
        "classifier": DecisionTreeClassifier,
        "name": "Gaussian Navie Bayes",
        "dir": "gnb",
        "args": {}
    },
    {
        "classifier": SVC,
        "name": "SVC",
        "dir": "svc",
        "args": {}
    }
]

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/')
RESULTS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../results/')
