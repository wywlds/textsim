import common
import numpy as np
import math
import tensorflow as tf
import metrics
vocab_size=2303
embedding_dim = 300
batch_size=10
num_epoch=10
hidden_state=200
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
    emb = tf.convert_to_tensor(common.getEmb(), dtype="float")
    (trainleft, trainright, trainscore) = common.getTrainSet()
    (testleft, testright, testscore) = common.getTestSet()
    (testlbatch, testllength,testrbatch, testrlength, testscores) = next_batch(len(testleft), testleft, testright, testscore, random=False)
    lenth = len(trainleft)
    embedding = emb
    W = tf.Variable(emb,
                    trainable=False, name="W")

    pivot = tf.placeholder(tf.float32, shape=[None, 5], name="pivot")
    leftseqs=tf.placeholder(tf.int32, shape=[None, None], name="leftseqs")
    leftlength=tf.placeholder(tf.float32, shape=[None], name="leftlength")
    rightseqs=tf.placeholder(tf.int32, shape=[None, None], name='rightseqs')
    rightlength=tf.placeholder(tf.float32, shape=[None], name="rightlength")
    leftEmbedding = tf.nn.embedding_lookup(W, leftseqs)
    rightEmbedding = tf.nn.embedding_lookup(W, rightseqs)

    leftSum = tf.reduce_sum(leftEmbedding, axis=1)
    leftAverage = tf.transpose(tf.multiply(tf.transpose(leftSum), leftlength))
    rightSum = tf.reduce_sum(rightEmbedding, axis=1)
    rightAverage = tf.transpose(tf.multiply(tf.transpose(rightSum), rightlength))

    product = tf.multiply(leftAverage, rightAverage)
    subs = tf.norm(tf.subtract(leftAverage, rightAverage), axis=1, keep_dims=True)
    W1 = tf.Variable(tf.random_uniform([hidden_state,300], -1.0, 1.0), name="W1")
    W2 = tf.Variable(tf.random_uniform([hidden_state,1], -1.0, 1.0), name="W2")
    bias = tf.Variable(tf.constant(0.1, shape=[hidden_state,1]), name="bias")
    w1p=tf.matmul(W1, tf.transpose(product))
    w2p = tf.matmul(W2, tf.transpose(subs))
    w3p = tf.add(w1p, w2p)
    w4p = tf.add(w3p, bias)
    ltransform = tf.transpose(tf.sigmoid(w4p))

    W3 = tf.Variable(tf.random_uniform([hidden_state, 5], -1.0, 1.0), name="W3")
    bias2 = tf.Variable(tf.constant(0.1, shape=[5]), name="bias2")
    projection = tf.nn.xw_plus_b(ltransform, W3, bias2)
    psoftmax = tf.nn.softmax(projection)
    value = tf.constant([[1.0], [2.0], [3.0], [4.0], [5.0]])
    prediction = tf.matmul(psoftmax, value)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=projection, labels=pivot)
    l2_regularizer = tf.contrib.layers.l2_regularizer(
        scale=0.0001, scope=None
    )
    weights = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
    losses = tf.reduce_mean(loss) + regularization_penalty
    train_op = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6).minimize(losses)
    #train_op = tf.train.GradientDescentOptimizer(0.03).minimize(losses)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        print "\n"
        for epoch in range(num_epoch):
            for i in range(lenth / batch_size):
                (left, llength, right,rlength, scores)=next_batch(batch_size, trainleft, trainright, trainscore)
                sess.run(train_op, feed_dict={leftseqs:left, rightseqs:right, leftlength:llength, rightlength:rlength, pivot:scores})
            transform_result = sess.run(prediction , feed_dict={leftseqs: testlbatch, rightseqs: testrbatch, leftlength:testllength, rightlength:testrlength, pivot: testscores})
            newScores = []
            for item in transform_result:
                newScores.append(item[0])
            print newScores
            # calibrated = metrics.calibration(newScores)
            # print calibrated
            metrics.evaluate(newScores,testscore)
        saver.save(sess, "../models/DAN.ckpt")