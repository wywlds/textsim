#coding=utf-8
from batcher import Batcher
import tensorflow as tf
from model import Model
import metrics
num_epoch = 50
batcher = Batcher()

def runEpoch(sess, model):
    model.assign_new_batch_size(sess, batcher.batch_size)
    numBatch = batcher.numOfBatchPerEpoch()
    for i in range(numBatch):
        (databatch1, masks1, databatch2, masks2, labelsbatch) = batcher.next_train_batch()
        feed_dict={
            model.input_data_s1:databatch1,
            model.input_data_s2:databatch2,
            model.mask_s1:masks1,
            model.mask_s2:masks2,
            model.target:labelsbatch
        }
        sess.run(model.train_op, feed_dict=feed_dict)

def evaluate(sess,testmodel):
    testmodel.assign_new_batch_size(sess, 4927)
    (databatch1, masks1, databatch2, masks2, labelsbatch) = batcher.next_test_batch()
    feed_dict = {
        testmodel.input_data_s1: databatch1,
        testmodel.input_data_s2: databatch2,
        testmodel.mask_s1: masks1,
        testmodel.mask_s2: masks2,
        testmodel.target: labelsbatch
    }
    result=sess.run(testmodel.prediction, feed_dict=feed_dict)
    newScores = []
    for item in result:
        newScores.append(item[0])
    print newScores
    scores=batcher.test_score()
    metrics.evaluate(newScores, scores)

def train(sess, model, testmodel):
    for i in range(num_epoch):
        runEpoch(sess, model)
        evaluate(sess, model)

def run():
    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=None):
            trainmodel = Model(sess=sess)
        with tf.variable_scope("model", reuse=True):
            testmodel = Model(sess=sess, is_Training=False, batch_size=4927)
        tf.global_variables_initializer().run()
        train(sess,trainmodel, testmodel)

if __name__=="__main__":
    run()

