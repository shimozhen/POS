
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
from chinese import ch_word_to_id

MAX_LINE_SIZE = 0
NUM_OF_FILES = 2448 #-1: ALL
RATIO = 0.95

test_output = "../data/output"
test_train = "../data/train"
test_test = "../data/test"

#padding and return x, y sequece
def sentence_handle(sentence, word_dict):
    global MAX_LINE_SIZE
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
    for i in range(length, MAX_LINE_SIZE):
        x.append("blank")
        y.append("blank")
    return x, y, length


def read_dir(file_directory, tag_dict, word_dict):
    cnt = 0
    NUM_OF_FILES = len(os.listdir(file_directory))
    for file in os.listdir(file_directory):
        if (file.endswith(".swp")):
            continue
        fin = open(file_directory + file, "r")

        cnt += 1
        for line in fin.readlines():

            endString = "（_PU 完_VV ）_PU"
            line = line.strip()
            if (endString in line or len(line) == 0 or line.startswith("<") or line.endswith(">")):
                continue
            x, y, length = sentence_handle(line, word_dict)
            if (length == 0):
                continue
            xx = []
            yy = []

            for word in x:
                xx.append(word_dict[word])
            for tag in y:
                yy.append(tag_dict[tag])
            append_write(test_output, xx, yy)
            if (cnt < int(NUM_OF_FILES * RATIO)):
                append_write(test_train, xx, yy)
            else:
                append_write(test_test, xx, yy)

def append_write(file, x, y):
    fout = open(file, "a")
    fout.write(("%d") % (x[0]))
    for i in range(1, len(x)):
        fout.write((" %d") % (x[i]))
    for item in y:
        fout.write((" %d") % (item))
    fout.write("\n")
    # fout.write(" ".join(str(x)) + " ")
    # fout.write(" ".join(str(y)) + "\n")

def insert(key, index, dict):
    if (key in dict.keys()):
        return index
    dict[key] = index
    index += 1
    return index

def length_tagsize_vacabsize(dir):
    maxlen = 0
    tag_set = set()
    char_set = set()
    tag_dict = {}
    word_dict = {}
    tag_index = 1
    word_index = 1
    for file in os.listdir(dir):
        if (file.endswith(".swp")):
            continue
        fin = open(dir + file, "r")
        for line in fin.readlines():
            endString = "（_PU 完_VV ）_PU"
            line = line.strip()
            if (endString in line or len(line) == 0 or line.startswith("<") or line.endswith(">")):
                continue
            length = 0
            word_tags = line.split(" ")
            for word_ in word_tags:
                if (len(word_.split("_")) != 2):
                    continue
                word, tag = word_.split("_")
                length += len(word)
                tag_set.add(tag)

                pre_tag = ["S", "B", "I", "E"]
                for pre in pre_tag:
                    tag_index = insert(pre + "_" + tag, tag_index, tag_dict)

                word_dict = ch_word_to_id
                # for char in word:
                #     char_set.add(char)
                #     word_index = insert(char, word_index, word_dict)

            maxlen = max(maxlen, length)

    tag_dict_file = open("tag_list", "w")
    for key in tag_dict.keys():
        tag_dict_file.write(("%s\t%s\n") % (key, tag_dict[key]))

    vacab_dict_file = open("vacab_list", "w")
    for key in word_dict.keys():
        vacab_dict_file.write(("%s\t%s\n") % (key, word_dict[key]))

    return maxlen, len(tag_set), len(word_dict.keys()), tag_dict, word_dict


def clear(path):
    fout = open(path, "w")
    fout.write("")
    fout.close()

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print("Usage: reader_ctb7.0 <input_dir> <output> <train> <test>")

    input_dir = "/home/shimozhen/NLP/joint_segmentation/ctb7.0/data/utf-8/postagged/"
    input_dir = sys.argv[1]
    test_output = sys.argv[2]
    test_train = sys.argv[3]
    test_test = sys.argv[4]

    # main(input_path, output_path)
    #read_dir("/home/shimozhen/NLP/joint_segmentation/ctb7.0/data/utf-8/postagged/")
    max_line_size, tag_size, vacab_size, tag_dict, word_dict = length_tagsize_vacabsize(input_dir)
    print("max_sequece_length:", max_line_size)
    print("distinct_tag_num:", tag_size * 4)
    print("vacab_size:", vacab_size)

    MAX_LINE_SIZE = max_line_size
    clear(test_output)
    clear(test_train)
    clear(test_test)
    word_dict["blank"] = 0
    tag_dict["blank"] = 0

    read_dir(input_dir, tag_dict, word_dict)

