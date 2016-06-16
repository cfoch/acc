import os
import time
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from settings import CLASSIFIERS_SETTINGS, DATA_DIR, RESULTS_DIR
from classifiers import classifier_from_settings
from sklearn.cross_validation import StratifiedKFold
from vectorizers import ACCVectorizer

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

class ACCReport:
    TEXT_REPORT_DIR = "text_report"
    WORDS_CLOUD_DIR = "word_clouds"
    BAR_CHART_DIR = "bar_chart"

    def __init__(self, documents, classes, tokenizer=None,
                 text_report=True, words_cloud=False, bar_chart=True):
        self.documents = documents
        self.classes = classes
        self.text_report = text_report
        self.words_cloud = words_cloud
        self.bar_chart = bar_chart
        self.tokenizer = tokenizer
        self.REPORTS_PATH = None

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        self.REPORTS_PATH = os.path.join(RESULTS_DIR,
            time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.REPORTS_PATH)

    def run(self):
        if self.REPORTS_PATH is None or self.tokenizer is None:
            return

        st = StratifiedKFold(self.classes, n_folds=4)

        i = 0
        for indTrain, indTest in st:
            current_report_path = os.path.join(self.REPORTS_PATH,
                "fold_%d" % (i + 1))

            tweets = self.documents[indTrain]
            sentiments = self.classes[indTrain]
            tweets_test = self.documents[indTest]
            sentiments_test = self.classes[indTest]

            acc_vectorizer = ACCVectorizer(
                tokenizer=self.tokenizer, stop_words='english')
            x = acc_vectorizer.fit_transform(tweets)
            y = sentiments

            x_test = acc_vectorizer.transform(tweets_test)
            y_test = sentiments_test

            for classifier_settings in CLASSIFIERS_SETTINGS:
                self.generate_report(current_report_path, classifier_settings,
                    x.toarray(), y, x_test.toarray(), y_test)
            i += 1

    def generate_report(self, base_path, classifier_settings,
                        x, y, xTest,yTest):
        print("Generating %s report." % classifier_settings["name"])
        path = os.path.join(base_path, classifier_settings["name"])
        os.makedirs(path)

        classifier = classifier_from_settings(classifier_settings)
        classifier.fit(x, y)
        yPred = classifier.predict(xTest)
        cm = confusion_matrix(yTest, yPred)
        precision, recall, fscore, support =\
            precision_recall_fscore_support(yTest, yPred, average='binary')
        accuracy = accuracy_score(yTest, yPred)

        if self.text_report:
            text_report_path = os.path.join(path, self.TEXT_REPORT_DIR)
            os.makedirs(text_report_path)
            with open(os.path.join(text_report_path, "report.txt"), "w") as f:
                self._simple_text_report(f, classifier_settings, precision,
                    recall, fscore, accuracy, cm)
        if self.words_cloud:
            pass
        if self.bar_chart:
            pass

    def _simple_text_report(self, f, classifier_settings, precision, recall,
                            fscore, accuracy, confusion_matrix=None):
        f.write("Classifier: %s\n" % classifier_settings["name"])
        f.write("precision: %lf\n" % precision)
        f.write("recall: %lf\n" % recall)
        f.write("fscore: %lf\n" % fscore)
        f.write("accuracy: %lf\n" % accuracy)
        f.write("confusion_matrix:\n")
        f.write("%s\n" % numpy.array_str(confusion_matrix, precision=4))
