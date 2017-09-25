import stats.idf as id
import re
global idfmap
import math
import metrics
def tfidf(left, right):
    wordsl = re.split(",| ", left)
    wordsr = re.split(",| ", right)
    rmap = {}
    for word in wordsr:
        word = word.lower()
        if word =="":
            continue
        if rmap.has_key(word):
            rmap[word]=rmap[word]+1
        else:
            rmap[word]=1
    lmap = {}
    for word in wordsl:
        word = word.lower()
        if word =="":
            continue
        if lmap.has_key(word):
            lmap[word]=lmap[word]+1
        else:
            lmap[word]=1
    product = 0.0
    lsquare = 0.0
    rsquare = 0.0
    for (word, num) in lmap.items():
        idf = idfmap[word]
        tfidf = float(num)*idf
        lsquare += tfidf * tfidf
        if rmap.has_key(word):
            rtf = rmap[word]
            product += tfidf * rtf * idf
    for (word, num) in rmap.items():
        idf = idfmap[word]
        tfidf = float(num)*idf
        rsquare += tfidf
    lroot = math.sqrt(lsquare)
    rroot = math.sqrt(rsquare)
    return product/(lroot*rroot)


if __name__=="__main__":
    idfmap = id.getIdf()
    testds = open("../dataset/sick/test.txt")
    tfidfs = []
    scores = []
    for l in testds.readlines():
        items = l.split("\t")
        id = items[0]
        sentenceA = items[1]
        sentenceB = items[2]
        score = float("%.1f" % float(items[3]))
        tfidfs.append(tfidf(sentenceA,sentenceB))
        scores.append(score)
    calibrated = metrics.calibration(tfidfs)
    print calibrated
    print scores
    metrics.evaluate(calibrated, scores)

    partitionScores = [[],[],[],[]]
    calibratedScores = [[],[],[],[]]
    for i in range(len(scores)):
        if scores[i] == 5.0:
            partitionScores[3].append(scores[i])
            calibratedScores[3].append(calibrated[i])
        else:
            position = int(scores[i])-1
            partitionScores[position].append(scores[i])
            calibratedScores[position].append(calibrated[i])
    print partitionScores[1][1:4]
    print calibratedScores[1][1:4]
    for i in range(4):
        metrics.evaluate(partitionScores[i], calibratedScores[i])