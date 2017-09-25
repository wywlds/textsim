import scipy.stats as st
import sklearn.metrics as skmetric
def pearson(x, y):
    return st.pearsonr(x, y)

def spearman(x, y):
    return st.spearmanr(x, y)

def mse(x, y):
    return skmetric.mean_squared_error(x,y)

def evaluate(x, y):
    print "r:%s rho:%s mse:%s"%(pearson(x,y), spearman(x,y), mse(x,y))

def mapAndInverseMap():
    trainingds = open("../dataset/sick/test.txt")
    scoremap={}
    def addscore(score):
        if scoremap.has_key(score):
            scoremap[score] = scoremap[score] + 1
        else:
            scoremap[score] = 1
    for l in trainingds.readlines():
        items = l.split("\t")
        score = float("%.1f"%float(items[3]))
        addscore(score)
    scoreList = list(scoremap.iteritems())
    scoreList.sort()
    x = []
    y = []
    for (score, num) in scoreList:
        x.append(score)
        y.append(num)
    #plotXY()
    sumnum = sum(y)
    ynormalized = []
    aggrenum = 0
    for num in y:
        aggrenum += num
        ynormalized.append(int(float(aggrenum)/float(sumnum)*100))
    dmap = {}
    imap = dict(zip(x, ynormalized))
    pos = 0
    for i in range(0,101):
        if i <= ynormalized[pos]:
            dmap[i] = x[pos]
        else:
            pos = pos + 1
            while ynormalized[pos]<i:
                pos = pos+1
            dmap[i]=x[pos]
    return (dmap, imap)

def calibration(seq):
    (dmap,_)=mapAndInverseMap()
    scores = []
    for i in seq:
        scores.append(dmap[int(st.percentileofscore(seq, i, kind="mean"))])
    return scores

if __name__=="__main__":
    (dmap, imap) = mapAndInverseMap()
    print dmap
    print imap