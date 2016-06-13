import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from settings import CLASSIFIERS_SETTINGS, DATA_DIR, RESULTS_DIR
from classifiers import classifier_from_settings


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
        print("support", support)
        print("=====================================================")
    return classifiers
