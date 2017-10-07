import numpy as np
from scipy import spatial
import metrics
from doc2vec import getinputs
import math
import tensorflow as tf

def next_batch(num, data1, data2, labels):
    idx = np.arange(0, len(data1))
    np.random.shuffle(idx)
    idx= idx[:num]
    data1batch = [data1[i] for i in idx]
    data2batch = [data2[i] for i in idx]
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
    return (data1batch, data2batch, labelsbatch)

def getSents(filename):
    vecs = np.fromfile(filename, dtype=np.float32)
    print vecs
    leftsent = []
    rightsent = []

    veclen = 4800
    for i in range(len(vecs)/veclen/2):
        leftsent.append(vecs[i * veclen * 2 : i * veclen * 2 + veclen])
        rightsent.append(vecs[i * veclen * 2 + veclen : (i + 1) * veclen * 2])
    print len(leftsent)
    return (leftsent, rightsent)

def productOnly(hidden_state):
    result = tf.multiply(x1, x2)
    W1 = tf.Variable(tf.random_uniform([hidden_state, input_dim], -1.0, 1.0), name="W1")
    bias = tf.Variable(tf.constant(0.1, shape=[hidden_state, 1]), name="bias")
    w1p = tf.matmul(W1, tf.transpose(result))
    w4p = tf.add(w1p, bias)
    ltransform = tf.transpose(w4p)
    print ltransform.shape
    return ltransform

if __name__=="__main__":
    (lefttest, righttest) = getSents("./test.skip")
    (lefttrain, righttrain) = getSents("./train.skip")
    (test1, test2, testtargets, originalScores) = getinputs.getTestInputs()
    (train1, traint2, traintargets, trainScores) = getinputs.getTrainInputs()

    input_dim = 4800
    hidden_state = 50
    num_epoch = 100
    batch_num=25
    x1=tf.placeholder(tf.float32, shape=[None, input_dim], name="x1")
    x2=tf.placeholder(tf.float32, shape=[None, input_dim], name="x2")
    pivot = tf.placeholder(tf.float32, shape=[None, 5], name="pivot")

    product = productOnly(hidden_state)
    subs = tf.abs(tf.subtract(x1, x2))

    W3 = tf.Variable(tf.random_uniform([hidden_state,5], -1.0, 1.0), name="W3")
    W4 = tf.Variable(tf.random_uniform([input_dim, 5], -1.0, 1.0), name="W4")
    bias2 = tf.Variable(tf.constant(0.1, shape=[5]), name="bias2")
    projection = tf.matmul(product, W3) + tf.matmul(subs, W4) + bias2

    psoftmax = tf.nn.softmax(projection)

    value = tf.constant([[1.0], [2.0], [3.0], [4.0], [5.0]])
    prediction = tf.matmul(psoftmax, value)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=projection, labels=pivot)

    l2_regularizer = tf.contrib.layers.l2_regularizer(
        scale=0.0001, scope=None
    )
    weights = tf.trainable_variables()  # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)

    losses = tf.reduce_mean(loss) + regularization_penalty
    train_op = tf.train.AdamOptimizer().minimize(losses)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(losses)
    #train_op = tf.train.AdadeltaOptimizer(1., 0.95, 1e-6).minimize(losses)
    init = tf.global_variables_initializer()

    lenth = len(lefttrain)

    with tf.Session() as sess:
        sess.run(init)
        print "\n"
        for epoch in range(num_epoch):
            for i in range(lenth/batch_num):
                (data1, data2, labels)=next_batch(batch_num, lefttrain, righttrain, trainScores)
                sess.run(train_op, feed_dict={x1: data1, x2: data2, pivot: labels})
            transform_result = sess.run(prediction , feed_dict={x1: lefttest, x2: righttest, pivot: [[0,0,0,0,0]]})
            newScores = []
            for item in transform_result:
                newScores.append(item[0])
            print newScores
            calibrated = metrics.calibration(newScores)
            metrics.evaluate(newScores,originalScores)

