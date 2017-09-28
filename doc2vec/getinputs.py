import gensim
import main
import metrics
def getTrainInputs():
    model = gensim.models.Doc2Vec.load("../dataset/sick/doc2vec")
    testdoc = open("../dataset/sick/train.txt")
    scores=[]
    inputs1 = []
    inputs2 = []
    originalScores=[]
    for l in testdoc.readlines():
        items = l.split("\t")
        sent1 = items[1]
        sent2 = items[2]
        words1 = main.simple_preprocess(sent1)
        words2 = main.simple_preprocess(sent2)
        inputs1.append(list(model.infer_vector(words1)))
        inputs2.append(list(model.infer_vector(words2)))
        score = float("%.1f" % float(items[3]))
        originalScores.append(score)
        scores.append((score - 1.0)/4)
    return (inputs1, inputs2, scores, originalScores)

def getTestInputs():
    model = gensim.models.Doc2Vec.load("../dataset/sick/doc2vec")
    testdoc = open("../dataset/sick/test.txt")
    scores=[]
    originalScores=[]
    inputs1 = []
    inputs2 = []
    for l in testdoc.readlines():
        items = l.split("\t")
        sent1 = items[1]
        sent2 = items[2]
        words1 = main.simple_preprocess(sent1)
        words2 = main.simple_preprocess(sent2)
        inputs1.append(list(model.infer_vector(words1)))
        inputs2.append(list(model.infer_vector(words2)))
        score = float("%.1f" % float(items[3]))
        originalScores.append(score)
        scores.append((score - 1.0)/4)
    return (inputs1, inputs2, scores, originalScores)

if __name__=="__main__":
    (inputs1, inputs2, targets) = getTrainInputs()
    print inputs1[:1]
    print inputs2[:1]
    print targets[:1]