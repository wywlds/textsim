import re
from gensim import corpora,models
def process():
    trainds = open("../dataset/sick/train.txt")
    testds = open("../dataset/sick/test.txt")
    texts = []
    def addSen(sent):
        words = re.split(",| ", sent)
        wordlist = []
        for word in words:
            if word == "":
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
    #dictionary.save('../dataset/sick/sick.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    #corpora.MmCorpus.serialize('../dataset/sick/sick.mm', corpus)
    #print dictionary.token2id
    model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
    print model.print_topics(2)

if __name__=="__main__":
    process()