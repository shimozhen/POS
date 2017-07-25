python3 -u ./gitrepo/model_test.py --train_dir ../data/vdata_nopunc/train \
	--test_dir ../data/vdata_nopunc/test \
	--log_dir ./log-freeze-embedding-no-punc-newdata/ \
	--c2v_path ../data/w2v/dnn_parser_vectors.char \
	--embedding_size 650 --batch_size 10 --hidden_size 200 --train_steps 1000 --learning_rate 0.001 --max_sentence_len 50
