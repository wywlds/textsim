import tensorflow as tf
import common
import numpy as np
emb = common.getEmb()
(trainleft, trainright, trainscore) = common.getTrainSet()
(testleft, testright, testscore) = common.getTestSet()
def average(sid):
    left = trainleft[sid]
    sum = np.zeros(300)
    for index in left:
        sum += emb[index]
    avg = sum/len(left)
    print avg
    print trainscore[sid]

def product(sid):
    left = testleft[sid]
    right = testright[sid]
    sumleft = np.zeros(300)
    sumright = np.zeros(300)
    for index in left:
        sumleft += emb[index]
    for index in right:
        sumright += emb[index]
    avgl = sumleft/len(left)
    avgr = sumright/len(right)
    print np.dot(avgl, avgr)

def restoreModel(filename):
    W = tf.get_variable("W", shape=[2303,300])
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, filename)
        print W.eval()

if __name__=="__main__":
    #restoreModel("../models/DAN.ckpt")
    product(0)
