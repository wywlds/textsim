import common
import numpy as np
import math
import tensorflow as tf
vocab_size=2303
embedding_dim = 300
batch_size=2
def next_batch(num, data1, data2, labels):
    idx = np.arange(0, len(data1))
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
    emb = tf.convert_to_tensor(common.getEmb(), dtype="float")
    print emb.shape
    (trainleft, trainright, trainscore) = common.getTrainSet()
    (testleft, testright, testscore) = common.getTestSet()
    embedding = emb
    W = tf.Variable(emb,
                    trainable=False, name="W")

    leftseqs=tf.placeholder(tf.int32, shape=[batch_size, None], name="leftseqs")
    leftlength=tf.placeholder(tf.float32, shape=[batch_size], name="leftlength")
    rightseqs=tf.placeholder(tf.int32, shape=[batch_size, None], name='rightseqs')
    rightlength=tf.placeholder(tf.float32, shape=[batch_size], name="rightlength")
    leftEmbedding = tf.nn.embedding_lookup(W, leftseqs)
    rightEmbedding = tf.nn.embedding_lookup(W, rightseqs)

    leftSum = tf.reduce_sum(leftEmbedding, axis=1)
    leftAverage = tf.transpose(tf.multiply(tf.transpose(leftSum), leftlength))
    rightSum = tf.reduce_sum(rightEmbedding, axis=1)
    rightAverage = tf.transpose(tf.multiply(tf.transpose(rightSum), rightlength))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        (left, llength, right,rlength, scores)=next_batch(batch_size, trainleft, trainright, trainscore)
        #print sess.run(x)
        print sess.run((leftSum, leftAverage, leftlength), feed_dict={leftseqs:left, rightseqs:right, leftlength:llength, rightlength:rlength})