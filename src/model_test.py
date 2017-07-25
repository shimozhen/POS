from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

root_dir = "/home/shimozhen/NLP/data/"
train_dir = root_dir + "train"
test_dir = root_dir + "test"
log_dir = root_dir + "default_log/"
c2v_path = root_dir + "dnn_parser_vectors.char"
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_dir", train_dir, "train file for test")
tf.app.flags.DEFINE_string("test_dir", test_dir, "test file for test")
tf.app.flags.DEFINE_string("c2v_path", c2v_path, "c2v path")
tf.app.flags.DEFINE_integer("max_sentence_len", 418, "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 650, "embedding size")
tf.app.flags.DEFINE_integer("batch_size", 20, "batch size")
tf.app.flags.DEFINE_integer("distinct_tag_num", 152, "tag num")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_string("log_dir", log_dir, "log directory")
tf.app.flags.DEFINE_integer("train_steps", 50000, "trainning steps")
tf.app.flags.DEFINE_integer("num_hidden", 200, "hidden unit size")
#messssssssssssssssssssssss
tf.app.flags.DEFINE_integer("vocab_size", 7500, "vocab size")
tf.app.flags.DEFINE_boolean("print_all", True, "print every batch in test data")

def load_data(path):
    x = []
    y = []

    fp = open(path, "r")
    for line in fp.readlines():
        line = line.strip()
        if not line:
            continue
        ss = line.split(" ")
        # embed()
        assert (len(ss) == (FLAGS.max_sentence_len * 2))
        lx = []
        ly = []

        for i in range(0, FLAGS.max_sentence_len):
            lx.append(int(ss[i]))
            ly.append(int(ss[i + FLAGS.max_sentence_len]))
        x.append(lx)
        y.append(ly)

    fp.close()
    return np.array(x), np.array(y)



class Model:
    def __init__(self, embeddingSize, distinctTagNum, c2vPath, numHidden):
        self.embeddingSize = embeddingSize
        self.distinctTagNum = distinctTagNum
        self.numHidden = numHidden
        self.c2v = self.load_w2v(c2vPath)
        self.words = tf.constant(self.c2v, name = "words")
        # self.words = tf.Variable(
        #     tf.truncated_normal([FLAGS.vocab_size, embeddingSize], -1.0, 1.0)
        # )
        with tf.variable_scope('Softmax') as scope:
            self.W = tf.get_variable(
                shape = [numHidden * 2, distinctTagNum],
                initializer = tf.truncated_normal_initializer(stddev=0.01),
                name = "weights",
                regularizer = tf.contrib.layers.l2_regularizer(0.001)
            )
            self.b = tf.Variable(tf.zeros([distinctTagNum], name = "bias"))
        self.transition_params = None

        self.inp = tf.placeholder(
            tf.int32,
            shape=[None, FLAGS.max_sentence_len],
            name="input_placeholder"
        )


    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        # embed()
        return length

    def inference(self, X, reuse=None):
        word_vectors = tf.nn.embedding_lookup(self.words, X)

        length = self.length(X)
        self.tp1 = tf.Print(word_vectors, [word_vectors], "word_vector")
        self.tp2 = tf.Print(length, [length], "length")
        length_64 = tf.cast(length, tf.int64)

        with tf.variable_scope("rnn_fwbw") as scope:
            forward_output, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.BasicLSTMCell(self.numHidden, reuse=reuse),
                word_vectors,
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_forward"
            )

            backward_output_, _ = tf.nn.dynamic_rnn(
                tf.contrib.rnn.BasicLSTMCell(self.numHidden, reuse=reuse),
                inputs = tf.reverse_sequence(word_vectors, length_64, seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backward"
            )

            backward_output = tf.reverse_sequence(backward_output_, length_64, seq_dim=1)
            # x = tf.unstack(word_vectors, axis = 1)
            # fw_cell = rnn.BasicLSTMCell(self.numHidden, reuse=reuse)
            # bk_cell = rnn.BasicLSTMCell(self.numHidden, reuse=reuse)
            # output_, forward_output, backward_output = rnn.static_bidirectional_rnn(fw_cell, bk_cell, inputs = x, dtype=tf.float32,
            #                                              sequence_length=length)

        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.numHidden * 2])

        # embed()
        matricized_unary_scores = tf.matmul(output, self.W) + self.b

        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, FLAGS.max_sentence_len, self.distinctTagNum]
        )
        return unary_scores, length

    def load_w2v(self, c2vPath):
        print("load w2v from " + c2vPath)
        f = open(c2vPath, "r")
        word_map = []
        for line in f.readlines():
            word_and_id, vector = line.strip().split("\t")
            word, id = word_and_id.split(" ")
            a = vector.split(" ")
            vector_list = []
            assert(FLAGS.embedding_size == len(a))
            for item in a:
                vector_list.append(float(item))
            word_map.append(vector_list)
        return word_map

    def test_unary_score(self):
        P, sequence_length = self.inference(self.inp, reuse=True)
        return P, sequence_length

    def loss(self, X, Y):
        P, sequence_length = self.inference(X)
        # embed()
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, Y, sequence_length
        )
        loss = tf.reduce_sum(-log_likelihood)
        return loss


def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)


def read_csv(batchsize, file_name):
    filename_queue = tf.train.string_input_producer([file_name])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, field_delim=" ", record_defaults=[[0] for i in range(FLAGS.max_sentence_len * 2)])
    ret = tf.train.shuffle_batch(decoded, batch_size=batchsize, capacity=batchsize * 50,
                            min_after_dequeue=batchsize)
    # embed()
    return ret

def inputs(path):
    whole = read_csv(FLAGS.batch_size, path)
    #embed()
    features = tf.transpose(tf.stack(whole[0:FLAGS.max_sentence_len]))
    label = tf.transpose(tf.stack(whole[FLAGS.max_sentence_len:]))
    return features, label

def test_evaluate(sess, unary_score, test_sequence_length, transMatrix, inp, tX, tY, step):
    total_equal = 0
    batch_size = FLAGS.batch_size
    total_len = tX.shape[0]
    batch_num = int((total_len - 1) / batch_size) + 1
    correct_labels = 0
    total_labels = 0

    aver_accuracy = 0
    for i in range(batch_num):
        endOff = (i + 1) * batch_size
        if (endOff > total_len):
            endOff = total_len
        y = tY[i * batch_size: endOff]

        feed_dict = {inp: tX[i * batch_size:endOff]}
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict
        )

        for tf_unary_scores_, y_, sequence_length_, input in zip(
            unary_score_val, y, test_sequence_length_val, tX
        ):
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            # embed()
            viterbi_squence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transMatrix)
            correct_labels += np.sum(np.equal(viterbi_squence, y_))
            total_labels += sequence_length_

        accuracy = 100.0 * correct_labels / float(total_labels)
        aver_accuracy += accuracy
        if (FLAGS.print_all):
            print("Step: %d Accuracy: %.2f%%" % (step, accuracy))
    aver_accuracy /= batch_num
    print("Average accuracy: %.2f%%" % (aver_accuracy))


def main(_):
    train_path = FLAGS.train_dir
    test_path = FLAGS.test_dir
    c2v_path = FLAGS.c2v_path
    print("train_path:", train_path)
    print("test_path:", test_path)
    print("c2v_path:", c2v_path)

    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.embedding_size, FLAGS.distinct_tag_num, c2v_path, FLAGS.num_hidden)
        X, Y = inputs(train_path)
        tX, tY = load_data(test_path)

        total_loss = model.loss(X, Y)
        train_op = train(total_loss)
        test_unary_score, test_sequence_length = model.test_unary_score()
        sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        # sv = tf.train.Supervisor(graph=graph)
        with sv.managed_session(master="") as sess:
            training_steps = FLAGS.train_steps
            for step in range(training_steps):
                if (sv.should_stop()):
                    break
                try:
                    _, trainsMatirics = sess.run([train_op, model.transition_params])
                    # sess.run(model.tp1)
                    # sess.run(model.tp2)

                    if (step % 20 == 0):
                        print("[%d] loss: [%r]" % (step, sess.run(total_loss)))
                    if (step % 100 == 0):
                        test_evaluate(
                            sess, test_unary_score, test_sequence_length,
                            trainsMatirics, model.inp, tX, tY,
                            step
                        )
                        sv.saver.save(sess, FLAGS.log_dir + "model", global_step=step)
                except KeyboardInterrupt as e:
                    sv.saver.save(sess, FLAGS.log_dir + "model", global_step=step + 1)
                    raise e

            sv.saver.save(sess, FLAGS.log_dir + "final-model")
            sess.close()


if __name__ == "__main__":
    tf.app.run()