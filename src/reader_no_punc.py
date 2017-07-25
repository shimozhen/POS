from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from chinese import ch_word_to_id
from IPython import embed

seg_punc = ['。', '，', '：', '；']
MAX_SENTENCE_LENGTH = 50
ONLY_SEG = False
SEG_DICT = {"S":1, "B":2, "E":3, "I":4}

def insert(key, index, dict):
    if (key in dict.keys()):
        return index
    dict[key] = index
    index += 1
    return index

def read_split(input_dir):
    sequences = []
    sequence = []
    tagss = []
    tags = []
    tag_dict = {}
    tag_index = 1
    for file in os.listdir(input_dir):
        if (file.endswith(".swp")):
            continue
        fin = open(input_dir + file, "r")
        for line in fin.readlines():
            endString = "（_PU 完_VV ）_PU"
            line = line.strip()
            if (endString in line or len(line) == 0 or line.startswith("<") or line.endswith(">")):
                continue
            word_tags = line.split(" ")
            for word_tag in word_tags:
                if (len(word_tag.split("_")) != 2):
                    continue
                word, tag = word_tag.split("_")
                pre_tag = ["S", "B", "I", "E"]
                for pre in pre_tag:
                    tag_index = insert(pre + "_" + tag, tag_index, tag_dict)

                if (word in seg_punc):
                    sequences.append(sequence)
                    sequence = []
                    tagss.append(tags)
                    tags = []
                    continue
                sequence.append(word)
                tags.append(tag)
            if (len(sequence) != 0):
                sequences.append(sequence)
                sequence = []
                tagss.append(tags)
                tags = []
    return sequences, tagss, tag_dict

def code(x_, dict):
    x = []
    for item in x_:
        x.append(dict[item])
    return x

def sentence_to_code(sentences, tagss, tag_dict):
    xx = []
    xx_code = []
    yy = []
    yy_code = []
    max_length = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        tags = tagss[i]
        x = []
        y = []
        for j in range(len(sentence)):
            word = sentence[j]
            tag_ = tags[j]
            for k in range(len(word)):
                chara = word[k]
                if not (chara in ch_word_to_id):
                    continue
                x.append(chara)
                if not ONLY_SEG:
                    if (len(word) == 1):
                        y.append("S_" + tag_)
                    else:
                        if (k == 0):
                            y.append("B_" + tag_)
                        elif (k == len(word) - 1):
                            y.append("E_" + tag_)
                        else:
                            y.append("I_" + tag_)
                else:
                    if (len(word) == 1):
                        y.append("S")
                    else:
                        if (k == 0):
                            y.append("B")
                        elif (k == len(word) - 1):
                            y.append("E")
                        else:
                            y.append("I")

        if (len(x) > max_length):
            max_length = len(x)
        if (len(x) == 0):
            continue
        if (len(x) <= MAX_SENTENCE_LENGTH):
            xx.append(x)
            xx_code.append(code(x, ch_word_to_id))
            yy.append(y)
            if (ONLY_SEG):
                yy_code.append(code(y, SEG_DICT))
            else:
                yy_code.append(code(y, tag_dict))

    print("max length: " + str(max_length))
    # embed()
    return xx_code, yy_code, MAX_SENTENCE_LENGTH

def padding_and_output(encode_sentences, encode_tags, max_length, path):
    for i in range(len(encode_sentences)):
        sentence = encode_sentences[i]
        tag = encode_tags[i]
        for j in range(len(sentence), max_length):
            sentence.append(0)
            tag.append(0)
        assert (len(sentence) == max_length)

    fout = open(path, "w")
    for i in range(len(encode_sentences)):
        fout.write(("%d") % (encode_sentences[i][0]))
        for j in range(1, max_length):
            fout.write((" %d") % (encode_sentences[i][j]))
        for j in range(max_length):
            fout.write((" %d") % (encode_tags[i][j]))
        fout.write("\n")

if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print("Usage: reader_ctb7.0 <input_dir> <output> <only seg>")
    input_dir = sys.argv[1]
    output_filename = sys.argv[2]
    ONLY_SEG = bool(int(sys.argv[3]))

    # input_dir = "../../data/ctb7.0/data/utf-8/postagged/"
    # output_filename = "word_tag_nopunc"
    print("length set to: ", MAX_SENTENCE_LENGTH)
    sentences_no_punc, tagss, tag_dict = read_split(input_dir)
    encode_sentences, encode_tagss, max_length = sentence_to_code(sentences_no_punc, tagss, tag_dict)
    padding_and_output(encode_sentences, encode_tagss, max_length, output_filename)
