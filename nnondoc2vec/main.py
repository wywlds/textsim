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

if __name__=="__main__":
    (inputs1, inputs2, targets, originalTraining) = getTrainInputs()
    (test1, test2, testtargets, originalScores) = getTestInputs()
    print targets
    print testtargets
    lenthtest = len(test1)
    print inputs1[:1]
    print inputs2[:1]
    print targets[:1]
    input_dim=200
    batch_num=10
    num_epoch = 50
    x1=tf.placeholder(tf.float32, shape=[None, input_dim], name="x1")
    x2=tf.placeholder(tf.float32, shape=[None, input_dim], name="x2")
    result = tf.multiply(x1, x2)
    pivot = tf.placeholder(tf.float32, shape=[None], name="pivot")

    W = tf.Variable(tf.random_uniform([input_dim, 1], -1.0, 1.0), name="W")
    bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")

    ltransform = tf.nn.xw_plus_b(result, W, bias)
    init = tf.global_variables_initializer()

    l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=0.005, scope=None
    )
    weights = tf.trainable_variables()  # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

    loss = tf.reduce_mean(tf.abs(pivot-tf.transpose(ltransform))) + regularization_penalty
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    lenth = len(inputs1)


    with tf.Session() as sess:
        sess.run(init)
        print "\n"
        for epoch in range(num_epoch):
            for i in range(lenth/batch_num):
                (data1, data2, labels)=next_batch(batch_num, inputs1, inputs2, originalTraining)
                sess.run(train_op, feed_dict={x1: data1, x2: data2, pivot: labels})
            (cost,w,transform_result) = sess.run((loss,W, ltransform) , feed_dict={x1: test1[:lenthtest], x2: test2[:lenthtest], pivot: originalScores[:lenthtest]})
            newScores = []
            for item in transform_result:
                newScores.append(item[0])
            calibrated = metrics.calibration(newScores)
            metrics.evaluate(calibrated,originalScores)

