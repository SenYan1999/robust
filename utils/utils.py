import torch
import os
import nlp
import logging
import h5py

from torch.utils.data import Dataset
from logging import handlers
from transformers import BertTokenizer, AlbertTokenizer
from transformers.data.processors.glue import ColaProcessor, Sst2Processor, MnliProcessor, MrpcProcessor, QnliProcessor, \
    QqpProcessor, WnliProcessor, RteProcessor
from transformers.data.processors.glue import glue_convert_examples_to_features
from tqdm import trange, tqdm
from collections import Counter
from datasets import load_dataset
from nltk import word_tokenize
from utils import get_mlm_data_from_tokens

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger

class GlueDataset(Dataset):
    def __init__(self, data_dir, task, max_len, bert_name, bert_type, mode='train'):
        self.task = task
        self.max_len = max_len
        if bert_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        elif bert_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(bert_name)
        else:
            raise Exception('Please enter the correct bert type.')

        self.data, self.num_class = self._get_data(data_dir, mode)

    def _get_data(self, data_dir, mode):
        # define processors
        processors = {'CoLA': ColaProcessor,
                      'SST-2': Sst2Processor,
                      'MNLI': MnliProcessor,
                      'MRPC': MrpcProcessor,
                      'QNLI': QnliProcessor,
                      'QQP': QqpProcessor,
                      'RTE': RteProcessor,
                      'WNLI': WnliProcessor}

        # get InputExamples from raw file
        p = processors[self.task]()
        if mode == 'train':
            input_examples = p.get_train_examples(data_dir=os.path.join(data_dir, self.task))
        elif mode == 'test':
            input_examples = p.get_dev_examples(data_dir=os.path.join(data_dir, self.task))
        else:
            raise Exception('mode must be in ["train", "dev"]...')

        # get InputFeatures from InputExamples
        input_features = glue_convert_examples_to_features(input_examples, tokenizer=self.tokenizer, \
                                                           max_length=self.max_len, task=self.task.lower())

        # convert InputFeatures to tensor
        input_ids, attention_mask, token_type_ids, labels = [], [], [], []
        for feature in input_features:
            input_ids.append(feature.input_ids)
            attention_mask.append(feature.attention_mask)
            token_type_ids.append(feature.token_type_ids)
            labels.append(feature.label)
        input_ids, attention_mask, token_type_ids, labels = map(lambda x: torch.LongTensor(x),
                                                                (input_ids, attention_mask, token_type_ids, labels))

        return (input_ids, attention_mask, token_type_ids, labels), len(p.get_labels())

    def __getitem__(self, item):
        out = ()
        for i in self.data:
            out += (i[item],)
        return out

    def __len__(self):
        return self.data[0].shape[0]

def get_cola_data(tokenizer, data_dir, mode='train'):
    p = ColaProcessor()
    if mode == 'train':
        data = p.get_train_examples(data_dir=os.path.join(data_dir, 'cola'))
    elif mode == 'dev':
        data = p.get_dev_examples(data_dir=os.path.join(data_dir, 'cola'))
    elif mode == 'test':
        data = p.get_test_examples(data_dir=os.path.join(data_dir, 'cola'))
    input_sent, type_id, label = [], [], []
    for line in data:
        sent = ['[CLS]'] + tokenizer.tokenize(line.text_a)
        input_sent.append(sent)
        type_id.append([0 for _ in range(len(sent))])
        label.append(line.label)
    return input_sent, type_id, label

def get_sst2_data(tokenizer, data_dir, mode='train'):
    p = Sst2Processor()
    if mode == 'train':
        data = p.get_train_examples(data_dir=os.path.join(data_dir, 'sst-2'))
    elif mode == 'dev':
        data = p.get_dev_examples(data_dir=os.path.join(data_dir, 'sst-2'))
    elif mode == 'test':
        data = p.get_test_examples(data_dir=os.path.join(data_dir, 'sst-2'))
    input_sent, type_id, label = [], [], []
    for line in data:
        sent = ['[CLS]'] + tokenizer.tokenize(line.text_a)
        input_sent.append(sent)
        type_id.append([0 for _ in range(len(sent))])
        label.append(line.label)
    return input_sent, type_id, label

data_process_map = {'cola': get_cola_data, 'sst-2': get_sst2_data}

class MLMDataset(Dataset):
    def __init__(self, data_dir, task, max_len, bert_name, mode='train'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.data_dir = data_dir
        self.mode = mode
        self.task = task
        self.max_len = max_len

        # process data
        self.data = self.process_raw_data()

    def process_raw_data(self):
        # get input sent from processor
        input_sent, type_id, label = data_process_map[self.task](self.tokenizer, self.data_dir, self.mode)

        # construct mask data
        mlm_input, mlm_pred_token, token_type, atten_mask = [], [], [], []
        vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        for sent, tid in tqdm(zip(input_sent, type_id), total=len(input_sent)):
            mlm_sent, mlm_label = get_mlm_data_from_tokens(sent, self.tokenizer, vocab)
            mask = [1 for i in range(len(mlm_sent))]

            # assert all the input is in the same length
            if len(mlm_sent) > self.max_len:
                mlm_input.append(mlm_sent[:self.max_len])
                mlm_pred_token.append(mlm_label[:self.max_len])
                token_type.append(tid[:self.max_len])
                atten_mask.append(mask[:self.max_len])
            else:
                length = len(mlm_sent)
                mlm_input.append(mlm_sent + [0 for _ in range(self.max_len - length)])
                mlm_pred_token.append(mlm_label + [-1 for _ in range(self.max_len - length)])
                token_type.append(tid + [0 for _ in range(self.max_len - length)])
                atten_mask.append(mask + [0 for _ in range(self.max_len - length)])
        
        # convert list to torch.LongTensor
        mlm_input, mlm_pred_token, token_type, atten_mask = torch.LongTensor(mlm_input), \
                                                            torch.LongTensor(mlm_pred_token), \
                                                            torch.LongTensor(token_type), \
                                                            torch.LongTensor(atten_mask)
        return (mlm_input, atten_mask, token_type, mlm_pred_token)
    
    def __getitem__(self, i):
        batch = ()
        for item in self.data:
            batch += (item[i], )
        return batch
    
    def __len__(self):
        return self.data[0].shape[0]

class SNLIDataset(Dataset):
    def __init__(self, data_dir, task, max_len, bert_name, bert_type, mode='train'):
        self.mode = mode
        if bert_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        elif bert_type == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained(bert_name)
        self.data = self.convert_data(max_len)
        self.num_class = 3

    def convert_data(self, max_len):
        data = nlp.load_dataset('snli')[self.mode]
        idxs, sents, seq_idxs, labels = [], [], [], []
        count = 0

        premise = data['premise']
        hypothesis = data['hypothesis']
        label_list = data['label']
        for i in trange(data.num_rows):
            sent1, sent2, label = premise[i], hypothesis[i], label_list[i]

            if label not in [0, 1, 2]:
                count = count + 1
                continue

            sent = '[CLS] ' + sent1 + ' [SEP] ' + sent2 + ' [SEP]'
            sent = self.tokenizer.tokenize(sent)

            if len(sent) < max_len:
                sent += ['[PAD]' for _ in range(max_len - len(sent))]
            else:
                sent = sent[:max_len]
            sent = self.tokenizer.convert_tokens_to_ids(sent)
            sent1 = self.tokenizer.tokenize(sent1)
            seq_idx = [0 for _ in range(len(sent1) + 1)] + [1 for _ in range(max_len - len(sent1) - 1)]
            
            sents.append(sent)
            seq_idxs.append(seq_idx)

            labels.append(label)
        sents, seq_idxs, labels = torch.LongTensor(sents), torch.LongTensor(seq_idxs), torch.LongTensor(labels)
        attention_mask = (sents != 0).long()
        print('Total Wrong Label %d' % count)
        return (sents, attention_mask, seq_idxs, labels)
    
    def __getitem__(self, item):
        out = ()
        for i in self.data:
            out += (i[item],)
        return out

    def __len__(self):
        return self.data[0].shape[0]

class BertDataset(Dataset):
    def __init__(self, filename):
        self.data = self.get_data(filename)
        self.max_pred_len = 20

    def get_data(self, filename):
        data = h5py.File(filename, 'r')
        input_ids = torch.LongTensor(data['input_ids'])
        input_mask = torch.LongTensor(data['input_mask'])
        segment_ids = torch.LongTensor(data['segment_ids'])
        masked_lm_positions = torch.LongTensor(data['masked_lm_positions'])
        masked_lm_ids = torch.LongTensor(data['masked_lm_ids'])
        next_sentences_labels = torch.LongTensor(data['next_sentence_labels'])

        return (input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentences_labels)

    def __getitem__(self, index):
        out = ()
        for item in self.data:
            out += (item[index],)
        
        input_ids, input_mask, segment_ids, masked_lm_position, masked_lm_ids, next_sentence_labels = out

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_len
        padded_mask_indices = (masked_lm_position == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_position[:index]] = masked_lm_ids[:index]

        return [input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels]
    
    def __len__(self):
        return self.data[0].shape[0]

class NormalDataset(Dataset):
    def __init__(self, dataset_name, max_len, mode='train', word2idx=None):
        self.mode = mode
        self.max_len = max_len
        self.dataset_name = dataset_name.lower()

        # build word2idx
        if word2idx != None:
            self.word2idx = word2idx
        else:
            self.word2idx = self.build_word2idx()

        self.data = self.get_data()

    def get_data(self):
        dataset = load_dataset('glue', self.dataset_name.lower())[self.mode]
        if self.dataset_name.lower() == 'cola':
            data = self.cola_processor(dataset, self.max_len)
        elif self.dataset_name.lower() == 'snli':
            pass############
        return data
    
    def cola_processor(self, dataset, max_len):
        all_sents = dataset['sentence']
        labels = dataset['label']

        print(len(all_sents))

        sents = []
        masks = []
        for sent in tqdm(all_sents):
            sent_idx = [self.word2idx.get(word, self.word2idx['<unk>']) for word in word_tokenize(sent)]
            sent_mask = [1. for _ in range(len(sent_idx))]
            
            if max_len > len(sent_idx):
                sent_idx += [self.word2idx.get('<pad>') for _ in range(max_len - len(sent_idx))]
                sent_mask += [0 for _ in range(max_len - len(sent_mask))]
            else:
                sent_idx = sent_idx[:max_len]
                sent_mask = sent_mask[:max_len]

            sents.append(sent_idx)
            masks.append(sent_mask)
        
        sents = torch.LongTensor(sents)
        masks = torch.LongTensor(masks)
        labels = torch.LongTensor(labels)

        return (sents, masks, labels)
    
    def snli_processor(self, dataset, max_len):
        all_sents = dataset['']
    
    def build_word2idx(self):
        dataset = load_dataset('glue', self.dataset_name.lower())
        all_words = Counter()

        for sent in dataset['train']['sentence']:
            all_words.update(word_tokenize(sent))
        
        word2idx = {'<pad>': 0, '<unk>': 1}
        for word, num in all_words.items():
            if num > 2:
                word2idx[word] = len(word2idx)
        
        return word2idx
    
    def __len__(self):
        return self.data[-1].shape[0]
    
    def __getitem__(self, idx):
        out = ()
        for i in self.data:
            out += (i[idx],)
        return out


if __name__ == '__main__':
    dataset = MLMDatasete('../glue_data/CoLA', 'cola', 120, 'bert-base-uncased', 'train')
