import re
from gensim import corpora,models
def process():
    trainds = open("../dataset/sick/train.txt")
    testds = open("../dataset/sick/test.txt")
    texts = []
    meaninglesswords = open("./meaninglesswords")
    killingwords = []
    for l in meaninglesswords.readlines():
        words = l.split(",")
        for word in words:
            killingwords.append(word.strip())
    print killingwords
    def addSen(sent):
        words = re.split(",| ", sent)
        wordlist = []
        for word in words:
            word = word.lower()
            if word == "" or word in killingwords:
                continue
            else:
                wordlist.append(word)
        texts.append(wordlist)
    for l in trainds.readlines():
        items = l.split("\t")
        addSen(items[1])
        addSen(items[2])
    for l in testds.readlines():
        items = l.split("\t")
        addSen(items[1])
        addSen(items[2])
    dictionary = corpora.Dictionary(texts)
    dictionary.save('../dataset/sick/sick.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('../dataset/sick/sick.mm', corpus)

if __name__=="__main__":
    process()