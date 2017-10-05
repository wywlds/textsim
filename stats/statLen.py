#coding=utf-8

import common
import math
import scipy.stats as st
import sklearn.metrics as skmetric
import matplotlib.pyplot as plt

danfilename="../dan_scores"
lstmfilename="../lstm_scores"
(testleft, testright, testscore) = common.getTestSet()
senlen = [(len(testl) + len(testr)) / 2 for (testl, testr) in zip(testleft, testright)]

def getTestSentences():
    fd = open("../dataset/sick/test.txt")
    lines=[]
    for l in fd.readlines():
        lines.append(l)
    return lines
lines = getTestSentences()

def getScores(filename):
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
    danscores = getScores(danfilename)
    lstmscores = getScores(lstmfilename)
    danerrors = getErrorsDividedByLength(danscores, testscore)
    danr = [error[0][0] for error in danerrors]
    lstmerrors = getErrorsDividedByLength(lstmscores, testscore)
    lstmr = [error[0][0] for error in lstmerrors]
    # print len(danerrors)
    # print len(lstmerrors)
    # x = range(4, 26, 2)
    # print len(x)
    #
    # plt.plot(x, danr, label="Average Network")
    # plt.plot(x, lstmr, label="LSTM-RNN")
    # plt.xlabel("sentence length")
    # plt.ylabel("pearson")
    # plt.legend()
    # plt.show()
    danerrors = [predict-score for (predict, score) in zip(danscores, testscore)]
    lstmerrors = [predict-score for (predict, score) in zip(lstmscores, testscore)]

    errors = [abs(predict-score) for (predict, score) in zip(danerrors, lstmerrors)]
    sorted = sorted(range(len(errors)), key=lambda i: errors[i])
    print sorted
    print "TOP 3:"
    for i in range(3):
        print i
        print lines[sorted[i]]
        print "%f_%f_%f"%(danscores[sorted[i]], lstmscores[sorted[i]], testscore[sorted[i]])

    print "BOTTOM 3:"
    for i in range(10):
        print i
        print lines[sorted[-1-i]]
        print "%f_%f_%f"%(danscores[sorted[-1-i]], lstmscores[sorted[-1-i]], testscore[sorted[-1-i]])
