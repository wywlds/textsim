from gensim import corpora, models, similarities
def lda():
    dictionary = corpora.Dictionary.load("../dataset/sick/sick.dict")
    corpus = corpora.MmCorpus("../dataset/sick/sick.mm")
    print corpus
    print dictionary
    model = models.LdaModel(corpus, id2word=dictionary, num_topics=100, passes=10)
    print model.print_topics(20)
    model.save("../dataset/sick/model.lda")

if __name__=="__main__":
    lda()