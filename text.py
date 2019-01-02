from data_io import DataReader, gen_embedding_from_file, read_tag_vocab
from config import config
from model import sequence_labeling
from tqdm import tqdm
from todo import evaluate 
import torch
from randomness import apply_random_seed
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self,item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def peek(self):
        if self.isEmpty():
            return -1
        return self.items[-1]
def getAcc(golden_list, predict_list):
    # recall
    pos_index_queue = Queue()
    neg_index_queue = Queue()
    true_pos_num = 0
    true_neg_num = 0
    pred_tag_num = 0
    for i in range (len(golden_list)):
        # precision
        for j in range(len(predict_list[i])):
            if predict_list[i][j] != 'O':
                pos_index_queue.enqueue(j)
            else:
                neg_index_queue.enqueue(j)
        #check pos
        while(not pos_index_queue.isEmpty()):
            entity_index_list = []
            has_next = True
            this = pos_index_queue.dequeue()
            entity_index_list.append(this)
            while (has_next):
                if this == pos_index_queue.peek() - 1:
                    this = pos_index_queue.dequeue()
                    entity_index_list.append(this)
                else:
                    entity_index_list.append(entity_index_list[-1]+1)
                    has_next = False
            pred_tag_num += 1
            #check tags if they are the same
            is_matched = True
            for e in entity_index_list:
                if e == len(golden_list[i]):
                    break
                if golden_list[i][e] != predict_list[i][e]:
                    is_matched = False
            if is_matched:
                true_pos_num += 1
        #check neg
        while(not neg_index_queue.isEmpty()):
            this = neg_index_queue.dequeue()
            pred_tag_num += 1
            if golden_list[i][this] == predict_list[i][this]:
                true_neg_num += 1
    return (true_neg_num+true_pos_num)/pred_tag_num

_config = config()
word_embedding, word_dict = gen_embedding_from_file(_config.word_embedding_file, _config.word_embedding_dim)
char_embedding, char_dict = gen_embedding_from_file(_config.char_embedding_file, _config.char_embedding_dim)
tag_dict = read_tag_vocab(_config.output_tag_file)
_config.nwords = len(word_dict)
_config.ntags = len(tag_dict)
_config.nchars = len(char_dict)

model = sequence_labeling(_config, word_embedding, char_embedding)
model.load_state_dict(torch.load(config.model_file))
test = DataReader(_config, _config.test_file, word_dict, char_dict, tag_dict, _config.batch_size, is_train=True)
model.eval()
optimizer = torch.optim.Adam(model.parameters())
pred_dev_ins, golden_dev_ins = [], []
reversed_tag_dict = {v: k for (k, v) in tag_dict.items()}
for batch_sentence_len_list, batch_word_index_lists, batch_word_mask, batch_char_index_matrices, batch_char_mask, batch_word_len_lists, batch_tag_index_list in test:
    pred_batch_tag = model.decode(batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices, batch_word_len_lists, batch_char_mask)
    pred_dev_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in zip(pred_batch_tag.data.tolist(), batch_sentence_len_list.data.tolist())]
    golden_dev_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in zip(batch_tag_index_list.data.tolist(), batch_sentence_len_list.data.tolist())]
print(getAcc(golden_dev_ins,pred_dev_ins))


                    
