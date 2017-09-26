from gensim import corpora, models, similarities
def ldatfidf(numt):
    dictionary = corpora.Dictionary.load("../dataset/sick/sick.dict")
    corpus = corpora.MmCorpus("../dataset/sick/sick.mm")
    tfidf = models.TfidfModel(corpus)
    corpustfidf = tfidf[corpus]
    print corpus
    print dictionary
    model = models.LdaModel(corpustfidf, id2word=dictionary, num_topics=numt, passes=10)
    print model.print_topics(20)
    model.save("../dataset/sick/modeltfidf.lda")

if __name__=="__main__":
    ldatfidf(100)