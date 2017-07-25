# Segmentation and POS Model for Chinese

This is a TensorFlow implementation of segmentation. Simply
take the problem as an sequence to sequence problem.

#### Required Dependencies

 * Python 3.5
 * NumPy
 * TensorFlow 1.1.0-rc2

#### Data

This project bases on CTB7.0 data set.
Others, you could have your own:
   * word embbeding file (send it to ./data/ directory)
   * word dict (send it to ./src/dict/ dicrectory)
   * tag dict (send it to ./src/dict/ dicrectory)

To initialize data, edit data path in make_data.sh and run:
```
./make_data.sh
```
Default parameters:
```
python3 src/reader_ctb7.0.py \
    ../data/ctb7.0/data/utf-8/postagged/ \
    ../data/ctb7.0_seg.train \
    ../data/ctb7.0_seg.dev \
    50 \
    0.05 \
```
#### Train
To train the model, run:
```
./train.sh
```
Default parameters:
```
python3 -u ./src/train.py --mode train\
    --train_dir ./data/ctb.train \
	--test_dir ./data/ctb.dev \
	--log_dir ./log/freeze-embedding-newdata/ \
	--c2v_path ./data/dnn_parser_vectors.char.20g \
	--embedding_size 650 --batch_size 10 --hidden_size 200 --train_steps 1000 --learning_rate 0.001 --max_sentence_len 50
```




Default hyper parameters for model:

Argument | Default | Description
--- | --- | ---
--max_sentence_len|418|max num of tokens per query
--embedding_size|650|embedding size
--batch_size|20|batch size
--learning_rate|0.001|learning rate
--train_steps|50000|trainning steps
--num_hidden|200|hidden unit size
--vocab_size|7500|vocab size

Default Data Params:

Argument | Default
--- | --- | ---
train_dir|./data/ctb.train
test_dir|./data/ctb.dev"
log_dir|./log/default_log/"
c2v_dir|./data/dnn_parser_vectors.char.20g"
dict_path|./src/dict/"
distinct_tag_num|152

#### Infer
Infer mode avaliable, to enter infer mode, you must set log_path.


Default:
```
#!/bin/bash
python3 -u ./src/train.py --mode infer \
    --train_dir ./data/ctb.train \
	--test_dir ./data/ctb.dev \
	--log_dir ./log/freeze-embedding-newdata/ \
	--c2v_path ./data/dnn_parser_vectors.char.20g \
	--embedding_size 650 --batch_size 10 --hidden_size 200 --train_steps 1000 --learning_rate 0.001 --max_sentence_len 50
```
