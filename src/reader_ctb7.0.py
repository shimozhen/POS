# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
from dict.chinese import ch_word_to_id

# Test data ratio
TEST_RATIO = 0.05
# path to save data
TRAIN_PATH = "./data/ctb.train"
TEST_PATH = "./data/ctb.dev"
DICT_PATH = "./src/dict/"


def sentence_handle(sentence, word_dict, max_length):
    '''
    split data, padding and return sequece(list type)
    
    :param sentence: a sentence
    :param word_dict: word index to check word
    :param max_length: max sequence length
    :return: word, tag, length
    '''
    sentence = sentence.strip()
    word_tags = sentence.split(" ")
    x = []
    y = []
    for i in range(len(word_tags)):
        if (len(word_tags[i].split("_")) != 2):
            continue
        word, tag_ = (word_tags[i]).split("_")
        if (not (word in word_dict.keys())):
            continue
        if len(word) == 1:
            x.append(word[0])
            tag = "S_" + tag_
            y.append(tag)
        else:
            i -= 1
            for j in range(len(word)):
                x.append(word[j])
                if j == 0:
                    tag = "B_" + tag_
                elif j == len(word) - 1:
                    tag = "E_" + tag_
                else:
                    tag = "I_" + tag_
                y.append(tag)
                i += 1
    length = len(x)
    assert(len(x) == len(y))
    for i in range(length, max_length):
        x.append("pad")
        y.append("pad")
    return x, y, length


def read_dir(file_directory, tag_dict, word_dict, max_length):
    '''
    read ctb7.0 file from a postagged file directory, and
    save to TRAIN_PATH, TEST_PATH
    
    :param file_directory: postagged file diretory
    :param tag_dict: path to save tag dict
    :param word_dict: path to save word dict
    :param max_length: max sequence length
    :return: None
    '''
    print("============ Reader start ==============")
    print("\tFrom: %s" % (file_directory))
    print("\tmax_length: %d" % (max_length))
    cnt = 0
    cnt_correct_length = 0
    train_count = 0
    test_count = 0
    for file in os.listdir(file_directory):
        if (file.endswith(".swp")):
            continue
        fin = open(file_directory + file, "r")
        for line in fin.readlines():
            end_string = "（_PU 完_VV ）_PU"
            line = line.strip()
            if (end_string in line or len(line) == 0 or line.startswith("<") or line.endswith(">")):
                continue
            cnt += 1
            x, y, length = sentence_handle(line, word_dict, max_length)
            if (length == 0):
                continue
            if (length > max_length):
                continue
            cnt_correct_length += 1
            xx = []
            yy = []
            for word in x:
                xx.append(word_dict[word])
            for tag in y:
                yy.append(tag_dict[tag])

            test_in_hundred = round(TEST_RATIO * 100)
            if (cnt % 100 > test_in_hundred):
                append_write(TRAIN_PATH, xx, yy)
                train_count += 1
            else:
                append_write(TEST_PATH, xx, yy)
                test_count += 1

    print("\tALL %d lines, seq_length < %d: %d lines" % (cnt, seq_length, cnt_correct_length))
    print("\tTest ratio: %f" % (TEST_RATIO))
    print("\tTrain data path: %s, %d lines wroted" % (TRAIN_PATH, train_count))
    print("\tTest data path: %s, %d lines wroted" % (TEST_PATH, test_count))
    print("\tmax_length: %d" % (max_length))
    print("======================================\n\n")


# append write data to a file
def append_write(file, x, y):
    fout = open(file, "a")
    fout.write(("%d") % (x[0]))
    for i in range(1, len(x)):
        fout.write((" %d") % (x[i]))
    for item in y:
        fout.write((" %d") % (item))
    fout.write("\n")


def get_data_param(dir, gen_vocab_dict=0):
    '''
        Now use given chinese word index 
        if gen_vocab_dict generate a new dict with the input data
    '''
    print("============ Data Param ==============")
    maxlen = 0
    tag_set = set()
    char_set = set()
    sentence_count = 0
    for file in os.listdir(dir):
        if (file.endswith(".swp")):
            continue
        fin = open(dir + file, "r")
        for line in fin.readlines():
            endString = "（_PU 完_VV ）_PU"
            line = line.strip()
            if (endString in line or len(line) == 0 or line.startswith("<") or line.endswith(">")):
                continue
            sentence_count += 1
            length = 0
            word_tags = line.split(" ")
            '''
                S: single charactor word
                B: begging of word
                I: inner a word
                E: end of word
            '''
            for word_ in word_tags:
                if (len(word_.split("_")) != 2):
                    continue
                word, tag = word_.split("_")
                length += len(word)
                pre_tag = ["S", "B", "I", "E"]
                for pre in pre_tag:
                    tag_set.add(pre + "_" + tag)
                for char in word:
                    char_set.add(char)
            maxlen = max(maxlen, length)

    tag_list = sorted(list(tag_set))
    tag_dict = dict(zip(tag_list, range(1, len(tag_list) + 1)))
    if (gen_vocab_dict):
        char_list = sorted(list(char_set))
        word_dict = dict(zip(char_list, range(1, len(char_list) + 1)))
    else:
        word_dict = ch_word_to_id

    # padding
    word_dict["pad"] = 0
    tag_dict["pad"] = 0

    # save
    tag_dict_file = open(DICT_PATH + "tag_index", "w")
    for key in tag_dict.keys():
        tag_dict_file.write(("%s\t%s\n") % (key, tag_dict[key]))
    print("\ttag index saved in %s" % (DICT_PATH + "tag_index"))

    vacab_dict_file = open(DICT_PATH + "word_index", "w")
    for key in word_dict.keys():
        vacab_dict_file.write(("%s\t%s\n") % (key, word_dict[key]))
    print("\tword index saved in %s" % (DICT_PATH + "word_index"))

    # print
    tag_size = len(tag_dict.keys())
    vocab_size = len(word_dict.keys())
    print("\tmax_sequece_length: ", maxlen)
    print("\tsentence_count: ", sentence_count)
    print("\tdistinct_tag_num: ", tag_size)
    print("\tvocab_size: ", vocab_size)
    print("======================================\n\n")
    return maxlen, sentence_count, tag_size, vocab_size, tag_dict, word_dict


# clear a file
def clear(path):
    fout = open(path, "w")
    fout.write("")
    fout.close()


if __name__ == "__main__":
    '''
        run this file with start.sh
    '''
    if (len(sys.argv) != 6):
        print("Usage: reader_ctb7.0 <input_dir> <train_datapath> <test_datapath> <seq_length> <test_ratio>")

    input_dir = sys.argv[1]
    test_train = sys.argv[2]
    test_test = sys.argv[3]
    seq_length = int(sys.argv[4])
    TEST_RATIO = float(sys.argv[5])

    max_line_size, sentence_count, tag_size, vocab_size, tag_dict, word_dict = get_data_param(input_dir)
    clear(TRAIN_PATH)
    clear(TEST_PATH)
    read_dir(input_dir, tag_dict, word_dict, seq_length)
