import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from settings import classifiers
from utils import filter_tweets_from_csv, tokenizer, generate_tdidf_matrix,\
    separateDataSet, generic_predict


from decisionTrees import decisionTrees


path = "data/training.1600000.processed.noemoticon.10.csv"
#path = "data/data.csv"
data = filter_tweets_from_csv(path)

matrix, classes = generate_tdidf_matrix(data)
print(matrix.shape)

x, y, xTest,yTest = separateDataSet(matrix,classes, 0.8)

print(x.shape, y.shape, xTest.shape, yTest.shape)

decisionTrees(x, y, xTest, yTest)


# Cada Clasificador por separado mejor

for klass in classifiers:
    classifier = klass()
    classifier.fit(x, y)
    yPred = generic_predict(classifier, x, y, xTest, yTest)
    if yPred is None:
        continue
    cm = confusion_matrix(yTest, yPred)
    precision, recall, fscore, support = precision_recall_fscore_support(yTest,
        yPred)
    print("=====================================")
    print("classifier: ", classifier)
    print("Confusion matrix:")
    print(cm)
    print("=====================================")


# Creo que no es necesario agregar una columna más con las clases.
"""
b = numpy.zeros((matrix.shape[0], matrix.shape[1] + 1))
b[:, : -1] = matrix
for i in range(matrix.shape[0]):
    matrix[i,-1] = classes[i]
print(b)
"""

