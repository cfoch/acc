from sklearn import tree


def decisionTrees(x, y, xTest, yTest):
    clf = tree.DecisionTreeClassifier()
    print("Training Decision Tree...")
    clf = clf.fit(x,y)
    print("End of training.")
    yPred = clf.predict(xTest)
    
    correct = wrong = 0
    print("prediccion", yPred)
    for i in range(yPred.shape[0]):
        if yPred[i] == yTest[i]:
            correct += 1
        else:
            wrong += 1
    print(correct,wrong)
    
