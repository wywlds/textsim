#coding=utf-8
import tensorflow as tf
import common
import batcher
class Model(object):
    formatf = tf.float32
    # LSTM hidden state的维数
    hidden_size = 50

    # embedding 的维数
    embed_dim=300

    # 最后做softmax的维数
    hidden_state = 50

    cell =None
    def singleCell(self, scope, cell="lstm", reuse=None):
        if self.cell!=None:
            print self.cell.name
            return (self.cell, None)
        if cell == 'gru':
            with tf.variable_scope("grucell" + scope, reuse=reuse, dtype=self.formatf):
                self.cell = tf.contrib.rnn.GRUCell(self.hidden_size, reuse = tf.get_variable_scope().reuse)
        else:
            with tf.variable_scope("lstmcell"+scope, reuse=reuse, dtype=self.formatf) as vs:
                self.cell=tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
                #print [v for v in tf.global_variables() if v.name.startswith(vs.name)]
                print vs.name
                print self.cell.name
        with tf.variable_scope("cell_init_state"+scope, reuse=reuse, dtype=self.formatf):
            cell_init_state=None
            #cell_init_state=cell.zero_state(self.batch_size,dtype=tf.float32)
        return (self.cell, cell_init_state)

    def rnn(self, x, scope, cell="lstm", reuse=None):
        with tf.name_scope('RNN_' + scope), tf.variable_scope('RNN_' + scope, dtype=self.formatf):
            (_, init_state) = self.singleCell(scope, cell=cell, reuse=reuse)
            outputs, states = tf.nn.dynamic_rnn(self.cell, x, initial_state=init_state, time_major=False, dtype=self.formatf)
        return outputs

    def assign_new_lr(self, session, lr_value):
        lr,_=session.run([self.lr, self._lr_update], feed_dict={self.new_lr:lr_value})
        return lr

    def assign_new_batch_size(self, session, batch_size_value):
        session.run(self._batch_size_update, feed_dict={self.new_batch_size: batch_size_value})

    def __init__(self, sess, is_Training=True, batch_size=25):
        #得到embedding
        emb = common.getEmb()
        W = tf.Variable(emb, trainable=False, name="wdict", dtype=self.formatf)

        #batch_size参数,在test时可以改变
        self.batch_size=tf.Variable(batch_size, dtype=tf.int32, trainable=False)
        self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
        self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)

        #Learning rate, 可以调整
        self.lr = tf.Variable(0.0, trainable=False, dtype=self.formatf)
        self.new_lr = tf.placeholder(self.formatf, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

        #数据的初始化
        self.input_data_s1 = tf.placeholder(tf.int32, [None, None])
        self.input_data_s2 = tf.placeholder(tf.int32, [None, None])
        self.embedding1 = tf.nn.embedding_lookup(W, self.input_data_s1)
        self.embedding2 = tf.nn.embedding_lookup(W, self.input_data_s2)
        self.target = tf.placeholder(self.formatf, [None, 5])
        self.mask_s1 = tf.placeholder(self.formatf, [None, None])
        self.mask_s2 = tf.placeholder(self.formatf, [None, None])

        with tf.name_scope('lstm_output_layer'):
            if is_Training:
                self.cell_outputs1 = self.rnn(x=self.embedding1, scope='side', cell='lstm', reuse=None)
                self.cell_outputs2 = self.rnn(x=self.embedding2, scope='side', cell='lstm', reuse=True)
            else:
                self.cell_outputs1 = self.rnn(x=self.embedding1, scope='side', cell='lstm', reuse=True)
                self.cell_outputs2 = self.rnn(x=self.embedding2, scope='side', cell='lstm', reuse=True)

        self.rawsent1 = tf.reduce_sum(self.cell_outputs1 * self.mask_s1[:, :, None], axis=1)
        self.rawsent2 = tf.reduce_sum(self.cell_outputs2 * self.mask_s2[:, :, None], axis=1)

        self.projectionW=tf.get_variable("projectionW",initializer=tf.random_uniform([self.hidden_state],-1.0,1.0,dtype=self.formatf))
        self.projectionR=tf.get_variable("projectionR",initializer=tf.random_uniform([self.hidden_state,1],-1.0,1.0,dtype=self.formatf))
        projection1 = tf.tensordot(self.cell_outputs1, self.projectionW,axes=1)+tf.tensordot(self.rawsent1, self.projectionR,axes=1)
        projection2 = tf.tensordot(self.cell_outputs2, self.projectionW,axes=1)+tf.tensordot(self.rawsent2, self.projectionR,axes=1)

        p1 = tf.exp(tf.nn.tanh(projection1)) * self.mask_s1[:,:]
        p2 = tf.exp(tf.nn.tanh(projection2)) * self.mask_s2[:,:]

        with tf.name_scope('Vector_Layer'):
            self.sent1 = tf.reduce_sum(self.cell_outputs1 * p1[:,:,None], axis=1) / tf.reduce_sum(p1[:,:,None], axis=1)
            self.sent2 = tf.reduce_sum(self.cell_outputs2 * p2[:,:,None], axis=1) / tf.reduce_sum(p2[:,:,None], axis=1)

        print self.cell_outputs1.shape
        with tf.name_scope('loss'):
            product = tf.multiply(self.sent1, self.sent2)
            subs = tf.abs(tf.subtract(self.sent1, self.sent2))
            W1 = tf.get_variable(initializer=tf.random_uniform([self.hidden_state, self.hidden_size], -1.0, 1.0, dtype=self.formatf), name="Wpro1",dtype=self.formatf)
            W2 = tf.get_variable(initializer=tf.random_uniform([self.hidden_state, self.hidden_size], -1.0, 1.0, dtype=self.formatf), name="Wpro2",dtype=self.formatf)
            self.bias = tf.get_variable(initializer=tf.constant(0.1, shape=[self.hidden_state, 1], dtype=self.formatf), name="bias",dtype=self.formatf)
            wp = tf.matmul(W1, tf.transpose(product))+tf.matmul(W2, tf.transpose(subs))+ self.bias
            ltransform = tf.transpose(tf.sigmoid(wp))

            W3 = tf.get_variable(initializer=tf.random_uniform([self.hidden_state, 5], -1.0, 1.0, dtype=self.formatf), name="Wpro3",dtype=self.formatf)
            bias2 = tf.get_variable(initializer=tf.constant(0.1, shape=[5], dtype=self.formatf), name="bias2",dtype=self.formatf)
            projection = tf.nn.xw_plus_b(ltransform, W3, bias2)
            psoftmax = tf.nn.softmax(projection)
            value = tf.constant([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=self.formatf)
            self.prediction = tf.matmul(psoftmax, value)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=projection, labels=self.target)
            l2_regularizer = tf.contrib.layers.l2_regularizer(
                scale=0.0001, scope=None
            )
            self.weights = tf.trainable_variables()
            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, self.weights)
            self.loss = tf.reduce_mean(loss) + regularization_penalty

        if not is_Training:
            return
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.05, epsilon=1e-6)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        with tf.name_scope('train'):
            self.train_op = optimizer.minimize(loss=self.loss)

