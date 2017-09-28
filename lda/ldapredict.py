from gensim import corpora, models, similarities
import re
import metrics
def predict(numt):
    lda = models.LdaModel.load("../dataset/sick/model.lda")
    #lda = models.LdaModel.load("../dataset/sick/modeltfidf.lda")
    dictionary = corpora.Dictionary.load("../dataset/sick/sick.dict")
    testds = open("../dataset/sick/test.txt")
    def splitSent(sent):
        words = re.split(",| ", sent)
        wordlist = []
        for word in words:
            if word == "":
                continue
            else:
                wordlist.append(word)
        return wordlist
    simscores=[]
    scores=[]
    for l in testds.readlines():
        items = l.split("\t")
        sent1 = items[1]
        txt1 = dictionary.doc2bow(splitSent(sent1))
        sent2 = items[2]
        txt2 = dictionary.doc2bow(splitSent(sent2))
        corpus = [txt1, txt2]
        index=similarities.MatrixSimilarity(lda[corpus],num_features=numt)
        sim = index[lda[txt2]]
        simscores.append(sim[0])

        score = float("%.1f" % float(items[3]))
        scores.append(score)
    calibrated = metrics.calibration(simscores)
    #print calibrated
    #print scores
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


if __name__=="__main__":
    predict(200)
