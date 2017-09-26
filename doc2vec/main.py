import gensim
import re
def simple_preprocess(sent):
    words = re.split(",| ", sent)
    wordlist = []
    for word in words:
        word = word.lower()
        if word == "":
            continue
        else:
            wordlist.append(word)
    return wordlist

def read_corpus(fname, tokens_only=False):
    with open(fname) as f:
        txtnum = 0
        for l in f.readlines():
            items = l.split("\t")
            sent1 = items[1]
            txtnum += 1
            if tokens_only:
                yield simple_preprocess(sent1)
            else:
                yield gensim.models.doc2vec.TaggedDocument(simple_preprocess(sent1), [txtnum])
            sent2 = items[2]
            txtnum += 1
            if tokens_only:
                yield simple_preprocess(sent2)
            else:
                yield gensim.models.doc2vec.TaggedDocument(simple_preprocess(sent2), [txtnum])
def getTrainCorpus():
    train_corpus = list(read_corpus("../dataset/sick/train.txt"))
    train_corpus+= list(read_corpus("../dataset/sick/test.txt"))
    return train_corpus
if __name__=="__main__":
    train_corpus = getTrainCorpus()
    #PV-DM
    model = gensim.models.doc2vec.Doc2Vec(dm=0,size=50, min_count=0, iter=100)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.save("../dataset/sick/doc2vec")
