#coding=utf-8
import tensorflow as tf
from doc2vec.getinputs import getTrainInputs,getTestInputs
import metrics
import numpy as np
def next_batch(num, data1, data2, labels):
    idx = np.arange(0, len(data1))
    np.random.shuffle(idx)
    idx= idx[:num]
    data1batch = [data1[i] for i in idx]
    data2batch = [data2[i] for i in idx]
    labelsbatch = [labels[i] for i in idx]
    return (data1batch, data2batch, labelsbatch)


def productSubNorm():
    result = tf.multiply(x1, x2)
    subs = tf.norm(tf.subtract(x1, x2), axis=1, keep_dims=True)
    print subs.shape
    W1 = tf.Variable(tf.random_uniform([1, input_dim], -1.0, 1.0), name="W1")
    W2 = tf.Variable(tf.random_uniform([1, 1], -1.0, 0), name="W2")
    bias = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")
    w1p = tf.matmul(W1, tf.transpose(result))
    w2p = tf.matmul(W2, tf.transpose(subs))
    w3p = tf.add(w1p, w2p)
    w4p = tf.add(w3p, bias)
    ltransform = tf.transpose(w4p)
    print ltransform.shape
    return ltransform

def productSub():
    result = tf.multiply(x1, x2)
    subs = tf.abs(tf.subtract(x1, x2))
    print subs.shape
    W1 = tf.Variable(tf.random_uniform([1, input_dim], -1.0, 1.0), name="W1")
    W2 = tf.Variable(tf.random_uniform([1, input_dim], -1.0, 0), name="W2")
    bias = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")
    w1p = tf.matmul(W1, tf.transpose(result))
    w2p = tf.matmul(W2, tf.transpose(subs))
    w3p = tf.add(w1p, w2p)
    w4p = tf.add(w3p, bias)
    ltransform = tf.transpose(w4p)
    print ltransform.shape
    return ltransform

def sub():
    subs = tf.abs(tf.subtract(x1, x2))
    W2 = tf.Variable(tf.random_uniform([1, input_dim], -1.0, 0), name="W2")
    bias = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")
    w2p = tf.matmul(W2, tf.transpose(subs))
    w4p = tf.add(w2p, bias)
    ltransform = tf.transpose(w4p)
    print ltransform.shape
    return ltransform

def productOnly():
    result = tf.multiply(x1, x2)
    W1 = tf.Variable(tf.random_uniform([1, input_dim], -1.0, 1.0), name="W1")
    bias = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")
    w1p = tf.matmul(W1, tf.transpose(result))
    w4p = tf.add(w1p, bias)
    ltransform = tf.transpose(w4p)
    print ltransform.shape
    return ltransform


if __name__=="__main__":
    (inputs1, inputs2, targets, originalTraining) = getTrainInputs()
    (test1, test2, testtargets, originalScores) = getTestInputs()
    print targets
    print testtargets
    lenthtest = len(test1)
    print inputs1[:1]
    print inputs2[:1]
    print targets[:1]
    input_dim=150
    batch_num=10
    num_epoch = 50

    #输入
    x1=tf.placeholder(tf.float32, shape=[None, input_dim], name="x1")
    x2=tf.placeholder(tf.float32, shape=[None, input_dim], name="x2")

    pivot = tf.placeholder(tf.float32, shape=[None], name="pivot")


    #ltransform=productSubNorm()
    #ltransform = productOnly()
    ltransform=productSub()
    #ltransform=sub()



    l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=0.005, scope=None
    )
    weights = tf.trainable_variables()  # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

    raw_loss = tf.losses.mean_squared_error([pivot], tf.sigmoid(ltransform))
    loss = raw_loss + regularization_penalty
    train_op = tf.train.AdadeltaOptimizer(learning_rate=0.05).minimize(loss)
    init = tf.global_variables_initializer()
    lenth = len(inputs1)


    with tf.Session() as sess:
        sess.run(init)
        print "\n"
        for epoch in range(num_epoch):
            for i in range(lenth/batch_num):
                (data1, data2, labels)=next_batch(batch_num, inputs1, inputs2, originalTraining)
                sess.run(train_op, feed_dict={x1: data1, x2: data2, pivot: labels})
            (cost,transform_result) = sess.run((loss, ltransform) , feed_dict={x1: test1[:lenthtest], x2: test2[:lenthtest], pivot: originalScores[:lenthtest]})
            newScores = []
            for item in transform_result:
                newScores.append(item[0])
            calibrated = metrics.calibration(newScores)
            metrics.evaluate(calibrated,originalScores)

