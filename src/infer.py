from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

root_dir = "/home/shimozhen/NLP/data/"
train_dir = root_dir + "train"
test_dir = root_dir + "test"
log_dir = "/home/shimozhen/NLP/joint_segmentation/log/"
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_dir", train_dir, "train file for test")
tf.app.flags.DEFINE_string("test_dir", test_dir, "test file for test")
tf.app.flags.DEFINE_integer("max_sentence_len", 418, "max num of tokens per query")
tf.app.flags.DEFINE_integer("embedding_size", 64, "embedding size")
tf.app.flags.DEFINE_integer("batch_size", 20, "batch size")
tf.app.flags.DEFINE_integer("distinct_tag_num", 152, "tag num")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_string("log_dir", log_dir, "log directory")
tf.app.flags.DEFINE_integer("train_steps", 50000, "trainning steps")
tf.app.flags.DEFINE_integer("num_hidden", 200, "hidden unit size")
tf.app.flags.DEFINE_integer("vocab_size", 4700, "vocab size")

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
        self.words = tf.Variable(
            tf.truncated_normal([FLAGS.vocab_size, embeddingSize], -1.0, 1.0)
        )
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
    batch_size = FLAGS.batch_size
    total_len = tX.shape[0]
    batch_num = int((total_len - 1) / batch_size) + 1
    correct_labels = 0
    total_labels = 0

    fout = open(FLAGS.log_dir + "sample", "w")
    fout.write("")
    fout.close()
    fout = open(FLAGS.log_dir + "sample", "a")
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
            viterbi_squence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transMatrix)
            correct_labels += np.sum(np.equal(viterbi_squence, y_))
            total_labels += sequence_length_

            fout.write(("length:%d sequence:%s\n") % (sequence_length_, list(input)[0:sequence_length_]))
            fout.write(("length:%d tags:%s\n") % (sequence_length_, viterbi_squence[0:sequence_length_]))

        accuracy = 100.0 * correct_labels / float(total_labels)
        print("Step: %d Accuracy: %.2f%%" % (step, accuracy))

def get_one(sess, unary_score, length, model, sentence):
    feed_dict = {model.inp: [sentence]}
    unary_score_val, length_val = sess.run([unary_score, length], feed_dict)
    [trans_matirics] = sess.run([model.transition_params])

    unary_score_val = unary_score_val[0][:int(length_val)]
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_score_val, trans_matirics)
    return viterbi_sequence

def get_tags(tag_encoded):
    tag_fin = open("tag_list")
    int_tag = {}
    for line in tag_fin.readlines():
        tag, code = line.strip().split("\t")
        int_tag[int(code)] = tag
    tags = []
    for item in tag_encoded:
        tags.append(int_tag[item])
    return tags

def get_encode_sentence(sentence):
    word_fin = open("vacab_list")
    word_int = {}
    for line in word_fin.readlines():
        word, int = line.strip().split("\t")
        word_int[word] = int
    ints = []
    for item in sentence:
        if (item in word_int.keys()):
            ints.append(word_int[item])
        else:
            ints.append(4699)
    for i in range(len(ints), FLAGS.max_sentence_len):
        ints.append(0)
    return ints

def create_tagged_sentence(sentence, tags):
    assert(len(sentence) == len(tags))
    tagged_words = []
    word = []
    tag = ""
    for i in range(len(tags)):
        seg, type = tags[i].split("_")
        if (seg == 'B'):
            word.clear()
            word.append(sentence[i])
            tag = type
        elif (seg == "I"):
            word.append(sentence[i])
            assert(tag == type, "I tag != B tag")
        elif (seg == "E"):
            word.append(sentence[i])
            assert(tag == type, "E tag != B tag")
            tagged_words.append("".join(word) + "/" + tag + " ")
        elif (seg == "S"):
            word.clear()
            word.append(sentence[i])
            tagged_words.append("".join(word) + "/" + type + " ")

    return "".join(tagged_words)


def main(_):
    train_path = "../../data/train_for_infer"
    test_path = "../../data/test_for_infer"
    raw_test_path = "../../data/raw_test_data"
    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.embedding_size, FLAGS.distinct_tag_num, "", FLAGS.num_hidden)
        X, Y = inputs(train_path)
        tX, tY = load_data(test_path)

        total_loss = model.loss(X, Y)
        # train_op = train(total_loss)
        test_unary_score, test_sequence_length = model.test_unary_score()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            saver.restore(sess, 'log/final-model')
            # sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
            # with sv.managed_session(master="") as sess:

            while(1):
                print("input a sentence:")
                test_sentence = input()
                if (test_sentence == ""):
                    print("blank sentence")
                    continue
                test_sentence_int = get_encode_sentence(test_sentence)
                res = get_one(sess, test_unary_score, test_sequence_length, model, test_sentence_int)
                tags = get_tags(res)
                print(tags)
                tagged_sentence = create_tagged_sentence(test_sentence, tags)
                print(tagged_sentence)

if __name__ == "__main__":
    tf.app.run()