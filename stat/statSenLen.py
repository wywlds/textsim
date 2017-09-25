from matplotlib import pyplot as plt
import re
from math import log
if __name__=="__main__":
    trainingds = open("../dataset/sick/train.txt")
    trainnum = 0
    lengthdict = {}
    def addSen(sentence):
        words = re.split(",| ", sentence)
        length = len(words)
        if lengthdict.has_key(length):
            lengthdict[length]=lengthdict[length]+1
        else:
            lengthdict[length]=1
    for l in trainingds.readlines():
        items = l.split("\t")
        id = items[0]
        sentenceA = items[1]
        addSen(sentenceA)
        sentenceB = items[2]
        addSen(sentenceB)
        score = items[3]
        entailment = items[4]
        trainnum += 1
    testds = open("../dataset/sick/test.txt")
    testnum = 0
    for l in testds.readlines():
        items = l.split("\t")
        id = items[0]
        sentenceA = items[1]
        addSen(sentenceA)
        sentenceB = items[2]
        addSen(sentenceB)
        score = items[3]
        entailment = items[4]
        testnum += 1
    print "training data set has %d pairs"%trainnum
    print "test data set has %d pairs" % testnum
    stats = list(lengthdict.iteritems())
    stats.sort()
    x = []
    y = []
    for (value, num) in stats:
        x.append(value)
        y.append(num)
    plt.bar(x,y)
    plt.xlabel("sentence length")
    plt.ylabel("num of instances")
    plt.show()