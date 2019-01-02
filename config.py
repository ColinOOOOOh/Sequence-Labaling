# You can change the paths/hyper-parameters in this file to experiment your implementation
class config:
	use_f1 = True
	use_char_embedding = False
	use_modified_LSTMCell = True

	train_file = 'C:/Users/ASUS/Desktop/6714/stage2/6714_spec_stage2/data/train.txt'
	dev_file = 'C:/Users/ASUS/Desktop/6714/stage2/6714_spec_stage2/data/dev.txt'
	test_file = 'C:/Users/ASUS/Desktop/6714/stage2/6714_spec_stage2/data/test.txt'
	output_tag_file = 'C:/Users/ASUS/Desktop/6714/stage2/6714_spec_stage2/data/tags.txt'
	char_embedding_file = 'C:/Users/ASUS/Desktop/6714/stage2/6714_spec_stage2/data/char_embeddings.txt'
	word_embedding_file = 'C:/Users/ASUS/Desktop/6714/stage2/6714_spec_stage2/data/word_embeddings.txt'
	model_file = 'C:/Users/ASUS/Desktop/6714/stage2/6714_spec_stage2/data/result_model_LSTM_f1.pt'

	word_embedding_dim = 50
	char_embedding_dim = 50
	char_lstm_output_dim = 50
	batch_size = 10
	hidden_dim = 50
	nepoch = 50
	dropout = 0.5

	nwords = 0
	nchars = 0
	ntags = 0