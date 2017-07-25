from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
'''
default paths 
You should use train.sh to run the code
'''
train_dir = "./data/ctb.train"
test_dir = "./data/ctb.dev"
log_dir = "./log/default_log/"
c2v_path = "./data/dnn_parser_vectors.char.20g"
dict_path = "./src/dict/"
FLAGS = tf.app.flags.FLAGS
'''
    hyper params and data features 
    send to model
'''
flags = tf.app.flags
flags.DEFINE_string("mode", "train", "train or infer")
flags.DEFINE_string("train_dir", train_dir, "train file for test")
flags.DEFINE_string("test_dir", test_dir, "test file for test")
flags.DEFINE_string("c2v_path", c2v_path, "c2v path")
flags.DEFINE_string("dict_path", dict_path, "dict path")
flags.DEFINE_integer("max_sentence_len", 418, "max num of tokens per query")
flags.DEFINE_integer("embedding_size", 650, "embedding size")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_integer("distinct_tag_num", 152, "tag num")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_string("log_dir", log_dir, "log directory")
flags.DEFINE_integer("train_steps", 50000, "trainning steps")
flags.DEFINE_integer("num_hidden", 200, "hidden unit size")
flags.DEFINE_integer("vocab_size", 7500, "vocab size")
flags.DEFINE_boolean("print_all", False, "print every batch in test data")
flags.DEFINE_boolean("continue_run", False, "continue run")


def load_data(path):
    '''
    load data to memory
    use for load test data, 
    
    :param path: path
    :return: sentences, tags
    '''
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


def inputs(path):
    '''
    input for train data, use queue in tensorflow 
    :param path: path
    :return: x, y
    '''
    batch_size = FLAGS.batch_size
    filename_queue = tf.train.string_input_producer([path])
    reader = tf.TextLineReader(skip_header_lines=0)
    key, value = reader.read(filename_queue)
    decoded = tf.decode_csv(value, field_delim=" ", record_defaults=[[0] for i in range(FLAGS.max_sentence_len * 2)])
    whole = tf.train.shuffle_batch(
        decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size
    )
    words = tf.transpose(tf.stack(whole[0:FLAGS.max_sentence_len]))
    tags = tf.transpose(tf.stack(whole[FLAGS.max_sentence_len:]))
    return words, tags


class Model:
    def __init__(self, embedding_size, distinct_tag_num, hidden_size, train_path, c2v_path):
        '''
        initial seq to seq model
        :param embedding_size:  
        :param distinct_tag_num: 
        :param c2v_path: 
        :param hidden_size: 
        :param train_path: train data path
        '''
        self.embeddingSize = embedding_size
        self.distinctTagNum = distinct_tag_num
        self.numHidden = hidden_size
        self.c2v = self.load_w2v(c2v_path)
        self.words = tf.constant(self.c2v, name="words")
        # random embedding
        # self.words = tf.Variable(
        #     tf.truncated_normal([FLAGS.vocab_size, embeddingSize], -1.0, 1.0)
        # )
        with tf.variable_scope('Softmax') as scope:
            # [2 * h, y]
            self.W = tf.get_variable(
                shape=[hidden_size * 2, distinct_tag_num],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="weights",
                regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )
            # [y]
            self.b = tf.Variable(tf.zeros([distinct_tag_num], name="bias"))
        self.transition_params = None
        # [?, x]
        self.infer_sentences = tf.placeholder(
            tf.int32,
            shape=[None, FLAGS.max_sentence_len],
            name="infer_sentence"
        )
        sentences, tags = inputs(train_path)
        self.loss = self.calc_loss(sentences, tags)
        self.unary_score, self.sequence_length = self.test_unary_score()
        self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    def length(self, data):
        '''
        calc length of padded batch data
        :param data: [b, x]
        :return: length [b]
        '''
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def inference(self, batch_word, reuse=None):
        '''
        forward and backward RNN cell, one layer 
        
        :param batch_word: batch of word
        :param reuse: 
        :return: potiential of tag, sequence length
        '''
        # [b, sequence_length, embedding_size]
        word_vectors = tf.nn.embedding_lookup(self.words, batch_word)
        length = self.length(batch_word)
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
                inputs=tf.reverse_sequence(word_vectors, length_64, seq_dim=1),
                dtype=tf.float32,
                sequence_length=length,
                scope="RNN_backward"
            )
            backward_output = tf.reverse_sequence(backward_output_, length_64, seq_dim=1)

        # [b, sequence_length, h * 2]
        output = tf.concat([forward_output, backward_output], 2)
        output = tf.reshape(output, [-1, self.numHidden * 2])
        matricized_unary_scores = tf.matmul(output, self.W) + self.b
        # [b, length, tag]
        unary_scores = tf.reshape(
            matricized_unary_scores,
            [-1, FLAGS.max_sentence_len, self.distinctTagNum]
        )
        return unary_scores, length

    def load_w2v(self, c2v_path):
        '''
        load word embedding
        :param c2v_path: path
        :return: a list storage word embeddings
        '''
        f = open(c2v_path, "r")
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
        '''
        a function returns potential and length, used for validation part
        reuse the parameters 
        :return: 
        '''
        potential, sequence_length = self.inference(self.infer_sentences, reuse=True)
        return potential, sequence_length

    def calc_loss(self, batch_word, batch_tag):
        '''
        train loss, used for train
        :param batch_word: batch of word 
        :param batch_tag: batch of tag
        :return: crf log likelihood loss
        '''
        potential, sequence_length = self.inference(batch_word, reuse=False)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            potential, batch_tag, sequence_length
        )
        loss = tf.reduce_sum(-log_likelihood)
        return loss


def test_evaluate(sess, model, trans_matrix, test_x, test_y, step):
    '''
    validation part of the model
    
    :param sess: session
    :param model: model 
    :param trans_matrix: train trans matrix which is trained  
    :param test_x: test x data
    :param test_y: test y data
    :param step: trainning step 
    :return: 
    '''
    print("--------- TEST -----------")
    batch_size = FLAGS.batch_size
    total_len = test_x.shape[0]
    batch_num = int((total_len - 1) / batch_size) + 1
    correct_labels = 0
    total_labels = 0
    aver_accuracy = 0

    for i in range(batch_num):
        end_off = (i + 1) * batch_size
        if (end_off > total_len):
            end_off = total_len
        x = test_x[i * batch_size:end_off]
        y = test_y[i * batch_size: end_off]

        feed_dict = {model.infer_sentences: x}
        unary_score, sequence_length = sess.run(
            [model.unary_score, model.sequence_length], feed_dict
        )

        for tf_unary_scores_, y_, sequence_length_ in zip(
            unary_score, y, sequence_length
        ):
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            viterbi_squence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, trans_matrix)
            correct_labels += np.sum(np.equal(viterbi_squence, y_))
            total_labels += sequence_length_

        accuracy = 100.0 * correct_labels / float(total_labels)
        aver_accuracy += accuracy
        if (FLAGS.print_all):
            print("Step: %d Accuracy: %.2f%%" % (step, accuracy))
    aver_accuracy /= batch_num
    print("[%d]\tAverage accuracy: %.2f%%" % (step, aver_accuracy))


'''
    infer part: 
    get_encode_sentence: sentence to index sentence 
    get_one: infer a sentence from model
    get_tags: index tags to tags
    create_tagged_sentence: make a tagged sentence 
'''


def get_one(sess, model, sentence):
    feed_dict = {model.infer_sentences: [sentence]}
    unary_score_val, length_val = sess.run([model.unary_score, model.sequence_length], feed_dict)
    [trans_matirics] = sess.run([model.transition_params])
    unary_score_val = unary_score_val[0][:int(length_val)]
    viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(unary_score_val, trans_matirics)
    return viterbi_sequence


def get_tags(tag_encoded, dict_path):
    tag_fin = open(dict_path + "tag_index")
    int_tag = {}
    for line in tag_fin.readlines():
        tag, code = line.strip().split("\t")
        int_tag[int(code)] = tag
    tags = []
    for item in tag_encoded:
        tags.append(int_tag[item])
    return tags


def get_encode_sentence(sentence, dict_path):
    word_fin = open(dict_path + "word_index")
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
        elif (seg == "E"):
            word.append(sentence[i])
            tagged_words.append("".join(word) + "/" + tag + " ")
        elif (seg == "S"):
            word.clear()
            word.append(sentence[i])
            tagged_words.append("".join(word) + "/" + type + " ")

    return "".join(tagged_words)


def main(_):
    # print
    train_path = FLAGS.train_dir
    test_path = FLAGS.test_dir
    c2v_path = FLAGS.c2v_path
    print("MODE: %s" % (FLAGS.mode))
    print("train_path:", train_path)
    print("test_path:", test_path)
    print("c2v_path:", c2v_path)

    graph = tf.Graph()
    with graph.as_default():
        model = Model(FLAGS.embedding_size, FLAGS.distinct_tag_num, FLAGS.num_hidden, train_path, c2v_path)
        test_x, test_y = load_data(test_path)
        if (FLAGS.continue_run):
            sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.log_dir)
        else:
            sv = tf.train.Supervisor(graph=graph)

        with sv.managed_session(master="") as sess:
            if (FLAGS.mode == "train"):
                training_steps = FLAGS.train_steps
                print("--------------------------\n"
                      "        TRAIN START       \n"
                      "       total step: %d     \n"
                      "--------------------------\n"
                      % (training_steps)
                      )
                for step in range(training_steps):
                    if (sv.should_stop()):
                        break
                    try:
                        _, trans_matrix = sess.run([model.optimizer, model.transition_params])
                        if (step % 5 == 0):
                            print("[%d] loss: [%r]" % (step, sess.run(model.loss)))
                        if (step % 100 == 0):
                            test_evaluate(
                                sess, model,
                                trans_matrix, test_x, test_y,
                                step
                            )
                            sv.saver.save(sess, FLAGS.log_dir + "model", global_step=step)
                    except KeyboardInterrupt as e:
                        sv.saver.save(sess, FLAGS.log_dir + "model", global_step=step + 1)
                        raise e

                sv.saver.save(sess, FLAGS.log_dir + "final-model")
            elif (FLAGS.mode == "infer"):
                print("--------------------------\n"
                      "        INFER START       \n"
                      "--------------------------\n"
                      )
                while (1):
                    print("input a sentence('exit' to exit):")
                    test_sentence = input()
                    if (test_sentence == ""):
                        print("blank sentence")
                        continue
                    if (test_sentence == "exit"):
                        break
                    test_sentence_int = get_encode_sentence(test_sentence, FLAGS.dict_path)
                    index_tags = get_one(sess, model, test_sentence_int)
                    tags = get_tags(index_tags, FLAGS.dict_path)
                    print(tags)
                    tagged_sentence = create_tagged_sentence(test_sentence, tags)
                    print(tagged_sentence)

            sess.close()


if __name__ == "__main__":
    tf.app.run()
