import numpy as np
import metrics
if __name__=="__main__":
    trainingds = open("../dataset/sick/train.txt")
    scores = []
    for l in trainingds.readlines():
        items = l.split("\t")
        score = float("%.1f" % float(items[3]))
        scores.append(score)

    testds = open("../dataset/sick/test.txt")
    testscores = []
    for l in testds.readlines():
        items = l.split("\t")
        score = float("%.1f" % float(items[3]))
        testscores.append(score)
    print np.average(scores)
    predictions = np.random.normal(np.average(scores), 0.01, len(testscores))
    print predictions
    metrics.evaluate(predictions, testscores)