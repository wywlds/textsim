import numpy
from scipy import spatial
import metrics
from doc2vec import getinputs

def getSents(filename):
    vecs = numpy.fromfile(filename, dtype=numpy.float32)
    print vecs
    leftsent = []
    rightsent = []

    veclen = 4800
    for i in range(len(vecs)/veclen/2):
        leftsent.append(vecs[i * veclen * 2 : i * veclen * 2 + veclen])
        rightsent.append(vecs[i * veclen * 2 + veclen : (i + 1) * veclen * 2])
    print len(leftsent)
    return (leftsent, rightsent)
if __name__=="__main__":
    (lefttest, righttest) = getSents("./test.skip")
    (lefttrain, righttrain) = getSents("./train.skip")
    (test1, test2, testtargets, originalScores) = getinputs.getTestInputs()
    cosines=[]
    for (left, right) in zip(lefttest, righttest):
        cosines.append(1- spatial.distance.cosine(left, right))
    calibrated = metrics.calibration(cosines)
    print len(calibrated)
    print originalScores
    metrics.evaluate(calibrated, originalScores)
