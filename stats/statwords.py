from matplotlib import pyplot as plt
import re
from math import log
if __name__=="__main__":
    trainingds = open("./dataset/sick/train.txt")
    trainnum = 0
    vocab = {}
    def addWord(sentence):
        words = re.split(",| ", sentence)
        for word in words:
            word = word.lower()
            if vocab.has_key(word):
                vocab[word]=vocab[word]+1
            else:
                vocab[word]=1
    for l in trainingds.readlines():
        items = l.split("\t")
        id = items[0]
        sentenceA = items[1]
        addWord(sentenceA)
        sentenceB = items[2]
        addWord(sentenceB)
        score = items[3]
        entailment = items[4]
        trainnum += 1
    print "vocab length:%d" % len(vocab)
    testds = open("./dataset/sick/test.txt")
    testnum = 0
    for l in testds.readlines():
        items = l.split("\t")
        id = items[0]
        sentenceA = items[1]
        addWord(sentenceA)
        sentenceB = items[2]
        addWord(sentenceB)
        score = items[3]
        entailment = items[4]
        testnum += 1
    print "training data set has %d pairs"%trainnum
    print "test data set has %d pairs" % testnum
    print "vocab length:%d"%len(vocab)

    statOnDF = {}
    def aggStat(value):
        if statOnDF.has_key(value):
            statOnDF[value]=statOnDF[value]+1
        else:
            statOnDF[value]=1
    plus500=[]
    minValue = 100000
    maxValue = 0

    for (key, value) in vocab.items():
        aggStat(value)
        if value>maxValue:
            maxValue = value
            maxKey = key
        if value<minValue:
            minValue = value
            minKey = key
        if value>500:
            plus500.append(key)
    print "maxValue:%d maxKey:%s"%(maxValue, maxKey)
    print "minValue:%d minKey:%s"%(minValue, minKey)
    print ",".join(plus500)

    stats = list(statOnDF.iteritems())
    stats.sort()
    stat2 = []
    x = []
    y = []
    for (value, num) in stats:
        if (value > 19854):
            x.append(0)
        else:
            x.append(log(19854/value))
        y.append(num)
    plt.plot(x, y)
    plt.xlabel("idf")
    plt.ylabel("Number of words")
    plt.show()

