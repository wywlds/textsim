import tensorflow as tf
from doc2vec.getinputs import getTrainInputs,getTestInputs
if __name__=="__main__":
    (inputs1, inputs2, targets) = getTrainInputs()
    (test1, test2, testtargets) = getTestInputs()
    lenthtest = len(test1)
    print inputs1[:1]
    print inputs2[:1]
    print targets[:1]
    input_dim=150
    batch_num=10
    num_epoch = 50
    x1=tf.placeholder(tf.float32, shape=[None, input_dim], name="x1")
    x2=tf.placeholder(tf.float32, shape=[None, input_dim], name="x2")
    pivot = tf.placeholder(tf.float32, shape=[None], name="pivot")
    sub = tf.subtract(x1, x2)
    W = tf.Variable(tf.random_uniform([input_dim, 1], -1.0, 1.0), name="W")
    bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
    ltransform = tf.nn.xw_plus_b(sub,W,bias)
    init = tf.global_variables_initializer()
    absvalue = tf.sigmoid(ltransform)
    loss = tf.reduce_mean(tf.abs(pivot-tf.transpose(absvalue)))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    lenth = len(inputs1)
    with tf.Session() as sess:
        sess.run(init)
        print "\n"
        for epoch in range(num_epoch):
            for i in range(len(inputs1)/batch_num):
                start = 10 * i
                end = 10 * (i + 1)
                sess.run(train_op, feed_dict={x1: inputs1[start:end], x2: inputs2[start:end], pivot: targets[start:end]})
            cost = sess.run(loss, feed_dict={x1: test1[:lenthtest], x2: test2[:lenthtest], pivot: testtargets[:lenthtest]})
            print "Epoch:%d loss:%f"%(epoch, cost)