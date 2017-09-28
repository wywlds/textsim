import tensorflow as tf
from doc2vec.getinputs import getTrainInputs,getTestInputs
import metrics
import numpy as np
import math
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
    result = tf.squeeze(tf.reduce_sum( tf.multiply( x1, x2 ), 1, keep_dims=True))
    subs = tf.norm(tf.subtract(x1,x2),axis=1)
    pivot = tf.placeholder(tf.float32, shape=[None, 5], name="pivot")

    W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="W1")
    W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="W2")
    bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")

    ltransform = tf.transpose([tf.sigmoid(W1*result + W2*subs + bias)])

    W3 = tf.Variable(tf.random_uniform([1,5], -1.0, 1.0), name="W3")
    bias2 = tf.Variable(tf.constant(0.1, shape=[5]), name="bias2")
    projection = tf.nn.xw_plus_b(ltransform, W3, bias2)
    psoftmax = tf.nn.softmax(projection)
    value = tf.constant([[1.0],[2.0],[3.0],[4.0],[5.0]])
    prediction=tf.matmul(psoftmax, value)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=projection, labels=pivot)
    init = tf.global_variables_initializer()

    l2_regularizer = tf.contrib.layers.l2_regularizer(
        scale=0.005, scope=None
    )
    weights = tf.trainable_variables()  # all vars of your graph
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)

    losses = tf.reduce_mean(loss) + regularization_penalty
    #losses = tf.reduce_mean(loss)

    train_op = tf.train.GradientDescentOptimizer(0.03).minimize(losses)
    lenth = len(inputs1)


    with tf.Session() as sess:
        sess.run(init)
        print "\n"
        for epoch in range(num_epoch):
            for i in range(lenth/batch_num):
                (data1, data2, labels)=next_batch(batch_num, inputs1, inputs2, originalTraining)
                sess.run(train_op, feed_dict={x1: data1, x2: data2, pivot: labels})
            transform_result = sess.run(prediction , feed_dict={x1: test1[:lenthtest], x2: test2[:lenthtest], pivot: [[0,0,0,0,0]]})
            newScores = []
            for item in transform_result:
                newScores.append(item[0])
            print newScores
            calibrated = metrics.calibration(newScores)
            metrics.evaluate(calibrated,originalScores)

