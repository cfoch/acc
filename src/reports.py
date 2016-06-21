import os
import time
import numpy
import matplotlib.pyplot as plt
import wordcloud

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from settings import CLASSIFIERS_SETTINGS, DATA_DIR, RESULTS_DIR
from classifiers import classifier_from_settings
from sklearn.cross_validation import StratifiedKFold
from vectorizers import ACCVectorizer

from IPython import embed

def plot_idf_stats(filename, features, idf, n=10):
    # Sort features and idf against idf and filter the
    # n elements with the higher idf.
    pairs = [(idf_tmp, feature) for idf_tmp, feature
             in sorted(zip(idf, features))]
    idf, features = zip(*pairs[-n:])
    idf, features = list(idf), list(features)
    idf.reverse()
    features.reverse()

    n = len(idf)
    x = numpy.arange(n)
    plt.bar(x, idf, align='center')
    plt.xticks(x, features)
    plt.savefig(filename)


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
            time.strftime("%b-%d-%I%M%p-%G"))
        os.makedirs(self.REPORTS_PATH)

    def run(self):
        if self.REPORTS_PATH is None or self.tokenizer is None:
            return

        st = StratifiedKFold(self.classes, n_folds=4)


        with open(os.path.join(self.REPORTS_PATH, "tokenizer_params.txt"),
                "w") as f:
            f.write(str(self.tokenizer.__dict__))


        if self.words_cloud:
            # Not available yet.
            self._generate_words_cloud_report()
        if not self.text_report and not self.bar_chart:
            return

        i = 0
        for indTrain, indTest in st:
            current_report_path = os.path.join(self.REPORTS_PATH,
                "fold_%d" % (i + 1))

            tweets = self.documents[indTrain]
            sentiments = self.classes[indTrain]
            tweets_test = self.documents[indTest]
            sentiments_test = self.classes[indTest]

            acc_vectorizer = ACCVectorizer(
                tokenizer=self.tokenizer.tokenize, stop_words='english')
            x = acc_vectorizer.fit_transform(tweets)
            y = sentiments

            x_test = acc_vectorizer.transform(tweets_test)
            y_test = sentiments_test

            if self.bar_chart:
                bar_chart_path = os.path.join(current_report_path,
                    self.BAR_CHART_DIR)
                os.makedirs(bar_chart_path)
                filename = os.path.join(bar_chart_path, "chart1.png")
                self._generate_chart_report(filename, acc_vectorizer)

            if self.text_report:
                for classifier_settings in CLASSIFIERS_SETTINGS:
                    self.generate_report(current_report_path,
                        classifier_settings, x.toarray(), y, x_test.toarray(),
                        y_test)
                    
            i += 1

    def generate_report(self, base_path, classifier_settings,
                        x, y, xTest, yTest):
        print("Generating %s report." % classifier_settings["name"])
        path = os.path.join(base_path, classifier_settings["dir"])
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
            with open(os.path.join(text_report_path, "params.txt"), "w") as f:
                f.write( str(classifier.get_params()))
            with open(os.path.join(text_report_path, "report.txt"), "w") as f:
                self._simple_text_report(f, classifier_settings, precision,
                    recall, fscore, accuracy, cm)

    def _simple_text_report(self, f, classifier_settings, precision, recall,
                            fscore, accuracy, confusion_matrix=None):
        f.write("Classifier: %s\n" % classifier_settings["name"])
        f.write("precision: %lf\n" % precision)
        f.write("recall: %lf\n" % recall)
        f.write("fscore: %lf\n" % fscore)
        f.write("accuracy: %lf\n" % accuracy)
        f.write("confusion_matrix:\n")
        f.write("%s\n" % numpy.array_str(confusion_matrix, precision=4))

    def _generate_chart_report(self, filename, vectorizer):
        print("Generating chart report.")
        plot_idf_stats(filename, vectorizer.get_feature_names(),
                       vectorizer._tfidf.idf_)

    def _generate_words_cloud_report(self):
        words_cloud_path = os.path.join(self.REPORTS_PATH, self.WORDS_CLOUD_DIR)
        os.makedirs(words_cloud_path)
        positives_path = os.path.join(words_cloud_path, "positives.png")
        negatives_path = os.path.join(words_cloud_path, "negatives.png")


        positive_documents = [d for (c, d) in zip(self.classes, self.documents)
            if c == 1]
        negative_documents = [d for (c, d) in zip(self.classes, self.documents)
            if c == 0]
        positive_vectorizer = CountVectorizer(tokenizer=self.tokenizer.tokenize,
            stop_words='english')
        negative_vectorizer = CountVectorizer(tokenizer=self.tokenizer.tokenize,
            stop_words='english')

        m_positive = positive_vectorizer.fit_transform(positive_documents)
        m_negative = negative_vectorizer.fit_transform(negative_documents)
        matrix_positive = m_positive.toarray()
        matrix_negative = m_negative.toarray()

        positives = [(tag, sum(row)) for (tag, row) in
            zip(positive_vectorizer.get_feature_names(), matrix_positive)]
        negatives = [(tag, sum(row)) for (tag, row) in
            zip(negative_vectorizer.get_feature_names(), matrix_negative)]

        w_positives = wordcloud.WordCloud()
        w_positives.generate_from_frequencies(positives)
        w_positives.to_file(positives_path)

        w_negatives = wordcloud.WordCloud()
        w_negatives.generate_from_frequencies(negatives)
        w_negatives.to_file(negatives_path)
