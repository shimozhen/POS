#!/bin/bash 
python3 -u ./src/train.py --mode train\
    --train_dir ./data/ctb.train \
	--test_dir ./data/ctb.dev \
	--log_dir ./log/freeze-embedding-newdata/ \
	--c2v_path ./data/dnn_parser_vectors.char.20g \
	--embedding_size 650 --batch_size 10 --hidden_size 200 --train_steps 1000 --learning_rate 0.001 --max_sentence_len 50
