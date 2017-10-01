#coding=utf-8
import common
import numpy as np
import math
class Batcher(object):
    batch_size=25
    (trainleft, trainright, trainscore) = common.getTrainSet()
    (testleft, testright, testscore) = common.getTestSet()

    def __init__(self):
        pass

    def numOfBatchPerEpoch(self):
        return len(self.trainright)/self.batch_size

    def next_train_batch(self):
        return self.next_batch(self.batch_size, self.trainleft, self.trainright, self.trainscore, random=True)

    def next_test_batch(self):
        return self.next_batch(len(self.testleft), self.testleft, self.testright, self.testscore, random=False)

    def test_score(self):
        return self.testscore

    def batch_one_side(self, data, idx):
        databatch=[]
        maxlength=0
        masks=[]
        for i in idx:
            maxlength = max(maxlength, len(data[i]))
        for i in idx:
            d=[]
            mask=[]
            senLen = len(data[i])
            for j in range(maxlength):
                mask.append(0)
                if j < senLen:
                    d.append(data[i][j])
                    if j == senLen - 1:
                        mask[j]=1
                else:
                    d.append(2302)
            databatch.append(d)
            masks.append(mask)
        return (databatch, masks)

    def bach_labels(self, scores, idx):
        labelsbatch = []
        for i in idx:
            label = np.zeros(5)
            sim = scores[i]
            ceil = math.ceil(sim)
            floor = math.floor(sim)
            if ceil == floor:
                label[int(ceil - 1)] = 1
            else:
                label[int(ceil - 1)] = sim - floor
                label[int(floor - 1)] = ceil - sim
            labelsbatch.append(label)
        return labelsbatch

    def next_batch(self, num, data1, data2, scores, random=True):
        idx = np.arange(0, len(data1))
        if random == True:
            np.random.shuffle(idx)
        idx = idx[:num]

        (databatch1, masks1) = self.batch_one_side(data1, idx)
        (databatch2, masks2) = self.batch_one_side(data2, idx)
        labelsbatch = self.bach_labels(scores, idx)
        return (databatch1, masks1, databatch2, masks2, labelsbatch)