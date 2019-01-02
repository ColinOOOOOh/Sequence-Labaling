import torch

from config import config
import torch.nn.functional as F

_config = config()

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

def evaluate(golden_list, predict_list):
        # recall
    pos_index_queue = Queue()
    pos_in_GT_num = 0
    found_pos_num = 0
    pos_in_pred_num = 0
    true_pos_num = 0
    for i in range (len(golden_list)):
        for j in range(len(golden_list[i])):
            if golden_list[i][j] != 'O':
                pos_index_queue.enqueue(j)
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
            pos_in_GT_num += 1
            #check tags if they are the same
            is_matched = True
            for e in entity_index_list:
                if e == len(golden_list[i]):
                    break
                if golden_list[i][e] != predict_list[i][e]:
                    is_matched = False
            if is_matched:
                found_pos_num += 1
        # precision
        for j in range(len(predict_list[i])):
            if predict_list[i][j] != 'O':
                pos_index_queue.enqueue(j)
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
            pos_in_pred_num += 1
            #check tags if they are the same
            is_matched = True
            for e in entity_index_list:
                if e == len(golden_list[i]):
                    break
                if golden_list[i][e] != predict_list[i][e]:
                    is_matched = False
            if is_matched:
                true_pos_num += 1
    recall = 0
    precision = 0
    f1 = 0
    if pos_in_GT_num == 0 :
        recall = 1
    else:
        recall = found_pos_num / pos_in_GT_num
    if pos_in_pred_num == 0:
        precision = 1
    else:
        precision = true_pos_num / pos_in_pred_num

    if (true_pos_num == 0 and (pos_in_pred_num > 0 or pos_in_GT_num > 0)):
        f1 = 0
        return f1
    elif (true_pos_num == 0 and pos_in_pred_num == 0 and pos_in_GT_num == 0):
        f1 = 1
        return f1
    else:
        f1 = (2*precision*recall) / (precision+recall)
        return f1




def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        hgates = F.linear(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (1-forgetgate) * cellgate
    hy = outgate * torch.tanh(cy)

    return hy, cy



def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    pass;

