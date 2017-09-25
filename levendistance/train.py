import levendistance
import metrics


def evaluateAll():
    testds = open("../dataset/sick/test.txt")
    levendistances = []
    scores = []
    for l in testds.readlines():
        splits = l.split("\t")
        sen1 = splits[1]
        sen2 = splits[2]
        score = float("%.1f" % float(splits[3]))
        scores.append(score)
        levendistances.append(-levendistance.leven(sen1,sen2))
    calibrated = metrics.calibration(levendistances)
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
    evaluateAll()