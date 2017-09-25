import re
from math import log
def getIdf():
    vocab = {}
    def addWord(sentence):
        words = re.split(",| ", sentence)
        curvocab={}
        for word in words:
            word = word.lower()
            if curvocab.has_key(word) or word=="":
                continue
            else:
                curvocab[word]=1

            if vocab.has_key(word):
                vocab[word] = vocab[word] + 1
            else:
                vocab[word] = 1

    trainingds = open("../dataset/sick/train.txt")
    for l in trainingds.readlines():
        items = l.split("\t")
        sentenceA = items[1]
        addWord(sentenceA)
        sentenceB = items[2]
        addWord(sentenceB)
    testds = open("../dataset/sick/test.txt")
    for l in testds.readlines():
        items = l.split("\t")
        sentenceA = items[1]
        addWord(sentenceA)
        sentenceB = items[2]
        addWord(sentenceB)

    idfs = {}
    for (word, num) in vocab.items():
        idfs[word]=log(19854 / num)
    return idfs

if __name__=="__main__":
    print getIdf()
    print len(getIdf())