import tensorflow as tf
import common
import numpy as np
import math
import metrics
from doc2vec.getinputs import getTrainInputs,getTestInputs

def next_batch(num, data1, data2, labels, random=True):
    idx = np.arange(0, len(data1))
    if random==True:
        np.random.shuffle(idx)
    idx= idx[:num]
    data1batch = []
    data2batch = []
    maxrlength = 0
    rlengths = []
    rinverse = []
    maxllength = 0
    llengths = []
    linverse = []
    for i in idx:
        maxllength = max(maxllength, len(data1[i]))
        maxrlength = max(maxrlength, len(data2[i]))
    for i in idx:
        llengths.append(len(data1[i]))
        linverse.append(1.0/len(data1[i]))
        rlengths.append(len(data2[i]))
        rinverse.append(1.0/len(data2[i]))
        d1=[]
        d2=[]
        for j in range(maxllength):
            if j < llengths[-1]:
                d1.append(data1[i][j])
            else:
                d1.append(2302)
        for j in range(maxrlength):
            if j < rlengths[-1]:
                d2.append(data2[i][j])
            else:
                d2.append(2302)
        data1batch.append(d1)
        data2batch.append(d2)
    labelsbatch = []
    for i in idx:
        label = np.zeros(5)
        sim = labels[i]
        ceil = math.ceil(sim)
        floor = math.floor(sim)
        if ceil == floor:
            label[int(ceil-1)]=1
        else:
            label[int(ceil-1)]=sim-floor
            label[int(floor - 1)]=ceil-sim
        labelsbatch.append(label)
    return (data1batch,linverse, data2batch,rinverse, labelsbatch)

if __name__=="__main__":
    input_dim=150

    emb = tf.convert_to_tensor(common.getEmb(), dtype="float")
    (trainleft, trainright, trainscore) = common.getTrainSet()
    (testleft, testright, testscore) = common.getTestSet()
    (test1, test2, testtargets, originalScores) = getTestInputs()
    (testlbatch, testllength, testrbatch, testrlength, testscores) = next_batch(len(testleft), testleft, testright,
                                                                                testscore, random=False)
    lenth = len(trainleft)
    embedding = emb
    W = tf.Variable(emb,
                    trainable=False, name="W")

    pivot = tf.placeholder(tf.float32, shape=[None, 5], name="pivot")
    leftseqs = tf.placeholder(tf.int32, shape=[None, None], name="leftseqs")
    leftlength = tf.placeholder(tf.float32, shape=[None], name="leftlength")
    rightseqs = tf.placeholder(tf.int32, shape=[None, None], name='rightseqs')
    rightlength = tf.placeholder(tf.float32, shape=[None], name="rightlength")
    leftEmbedding = tf.nn.embedding_lookup(W, leftseqs)
    rightEmbedding = tf.nn.embedding_lookup(W, rightseqs)

    x1=tf.placeholder(tf.float32, shape=[None, input_dim], name="x1")
    x2=tf.placeholder(tf.float32, shape=[None, input_dim], name="x2")

    leftSum = tf.reduce_sum(leftEmbedding, axis=1)
    leftAverage = tf.transpose(tf.multiply(tf.transpose(leftSum), leftlength))
    rightSum = tf.reduce_sum(rightEmbedding, axis=1)
    rightAverage = tf.transpose(tf.multiply(tf.transpose(rightSum), rightlength))

    init = tf.global_variables_initializer()
    product1 = tf.reduce_sum(tf.multiply(leftAverage, rightAverage), axis=1,keep_dims=True)/tf.norm(leftAverage, axis=1, keep_dims=True) / tf.norm(rightAverage, axis=1, keep_dims=True)
    product2 = tf.reduce_sum(tf.multiply(x1, x2), axis=1,keep_dims=True)/tf.norm(x1, axis=1, keep_dims=True) / tf.norm(x2, axis=1, keep_dims=True)
    product = product1 + product2
    print tf.norm(leftAverage, axis=1).shape
    print product.shape


    with tf.Session() as sess:
        sess.run(init)
        print "\n"
        # for epoch in range(num_epoch):
        #     for i in range(lenth / batch_size):
        #         (left, llength, right,rlength, scores)=next_batch(batch_size, trainleft, trainright, trainscore)
        #         sess.run(train_op, feed_dict={leftseqs:left, rightseqs:right, leftlength:llength, rightlength:rlength, pivot:scores})
        transform_result = sess.run(product , feed_dict={x1: test1, x2: test2,leftseqs: testlbatch, rightseqs: testrbatch, leftlength:testllength, rightlength:testrlength, pivot: testscores})
        newScores = []
        for item in transform_result:
            newScores.append(item[0])
        print newScores
        calibrated = metrics.calibration(newScores)
        # print calibrated
        metrics.evaluate(calibrated,testscore)
