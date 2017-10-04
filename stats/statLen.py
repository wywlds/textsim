import common
import math
import scipy.stats as st
import sklearn.metrics as skmetric

filename="../dan_scores"
(testleft, testright, testscore) = common.getTestSet()
senlen = [(len(testl) + len(testr)) / 2 for (testl, testr) in zip(testleft, testright)]

def getTestSentences():
    fd = open("../dataset/sick/test.txt")
    lines=[]
    for l in fd.readlines():
        lines.append(l)
    return lines
lines = getTestSentences()

def getScores():
    fd = open(filename)
    for l in fd.readlines():
        scores = l.split(",")
        scoresf = [float(score) for score in scores]
    return scoresf

def getErrorsDividedByLength(predict, test):
    newscores = [[] for i in range(13)]
    oldscores = [[] for i in range(13)]
    for i, lent in enumerate(senlen):
        if lent >= 24:
            newscores[12].append(predict[i])
            oldscores[12].append(test[i])
        else:
            newscores[lent/2].append(predict[i])
            oldscores[lent/2].append(test[i])
    errors = [(st.pearsonr(newscore, oldscore), st.spearmanr(newscore, oldscore),
               skmetric.mean_squared_error(newscore, oldscore)) for
              (newscore, oldscore) in zip(newscores, oldscores) if len(newscore) > 1]
    return errors


if __name__=="__main__":
    scores = getScores()
    errors = [predict-score for (predict, score) in zip(scores, testscore)]
    sorted = sorted(range(len(errors)), key=lambda i: errors[i])
    print sorted
    print "TOP 3:"
    for i in range(10):
        print lines[sorted[i]]
        print scores[sorted[i]]

    print "BOTTOM 3:"
    for i in range(30):
        print errors[sorted[-1-i]]
        print lines[sorted[-1-i]]
        print scores[sorted[-1-i]]
    #print sorted(range(len(errors)), key=lambda i: errors[i])[:3]
