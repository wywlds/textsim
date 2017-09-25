from matplotlib import pyplot as plt
import re
from math import log


def plotXY(x, y):
    plt.bar(x, y, width=0.1)
    plt.xlabel("similarity score")
    plt.ylabel("percentile")
    plt.show()


if __name__=="__main__":
    trainingds = open("../dataset/sick/train.txt")
    scoremap={}
    def addscore(score):
        if scoremap.has_key(score):
            scoremap[score] = scoremap[score] + 1
        else:
            scoremap[score] = 1
    for l in trainingds.readlines():
        items = l.split("\t")
        score = float("%.1f"%float(items[3]))
        addscore(score)
    testds = open("../dataset/sick/test.txt")
    for l in testds.readlines():
        items = l.split("\t")
        score = float("%.1f"%float(items[3]))
        addscore(score)
    scoreList = list(scoremap.iteritems())
    scoreList.sort()
    print scoreList
    x = []
    y = []
    for (score, num) in scoreList:
        x.append(score)
        y.append(num)
    #plotXY()
    sumnum = sum(y)
    print sumnum
    ynormalized = []
    aggrenum = 0
    for num in y:
        aggrenum += num
        ynormalized.append(int(float(aggrenum)/float(sumnum)*100))
    plotXY(x, ynormalized)