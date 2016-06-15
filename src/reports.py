import os
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from settings import CLASSIFIERS_SETTINGS, DATA_DIR, RESULTS_DIR
from classifiers import classifier_from_settings

from IPython import embed

def plot_idf_stats(features, idf, n=10):
    # Sort features and idf against idf and filter the
    # n elements with the higher idf.
    pairs = [(idf_tmp, feature) for idf_tmp, feature
             in sorted(zip(idf, features))]
    idf, features = zip(*pairs[:n])
    idf = list(idf)
    print(idf, features)
    n = len(idf)
    x = numpy.arange(n)
    plt.bar(x, idf, align='center')
    plt.xticks(x, features)
    plt.savefig('fig.png')


def text_report(x, y, xTest, yTest):
    classifiers = []
    for classifier_settings in CLASSIFIERS_SETTINGS:
        classifier = classifier_from_settings(classifier_settings)
        classifier.fit(x, y)
        yPred = classifier.predict(xTest)
        cm = confusion_matrix(yTest, yPred)
        precision, recall, fscore, support =\
            precision_recall_fscore_support(yTest, yPred, average='binary')
        classifiers.append(classifier)
        print("=====================================================")
        print("classifier: ", classifier_settings["name"])
        print("Confusion matrix:")
        print(cm)
        print("precision", precision)
        print("recall", recall)
        print("fscore", fscore)
        print("accuracy", accuracy_score(yTest, yPred))
        print("=====================================================")
    return classifiers

