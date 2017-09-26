import gensim
import main
import metrics
from scipy import spatial
if __name__=="__main__":
    model = gensim.models.Doc2Vec.load("../dataset/sick/doc2vec")
    testdoc = open("../dataset/sick/test.txt")
    cosines=[]
    scores=[]
    for l in testdoc.readlines():
        items = l.split("\t")
        sent1 = items[1]
        sent2 = items[2]
        words1 = main.simple_preprocess(sent1)
        words2 = main.simple_preprocess(sent2)
        vec1 = list(model.infer_vector(words1))
        vec2 = list(model.infer_vector(words2))
        cosines.append(1- spatial.distance.cosine(vec1, vec2))
        score = float("%.1f" % float(items[3]))
        scores.append(score)
    calibrated = metrics.calibration(cosines)
    print calibrated
    print scores
    metrics.evaluate(calibrated, scores)
    # traincorpus = main.getTrainCorpus()
    # ranks = []
    # second_ranks = []
    # for i in range(len(traincorpus)):
    #     inferred_vector = model.infer_vector(traincorpus[i].words)
    #     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    #     rank = [docid for docid, sim in sims].index(i)
    #     ranks.append(rank)
    #     print rank
    #     second_ranks.append(sims[1])
    # collections.Counter(ranks)