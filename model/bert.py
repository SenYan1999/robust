import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, AlbertModel, BertConfig
from utils.mlm import get_mask_subset_with_prob, mask_with_tokens, prob_mask_like

class Bert(nn.Module):
    def __init__(self, bert_name, num_class, bert_type='bert', drop_out=0.1):
        super(Bert, self).__init__()
        if bert_type == 'bert':
            self.bert = BertModel.from_pretrained(bert_name)
        elif bert_type == 'albert':
            self.bert = AlbertModel.from_pretrained(bert_name)
        else:
            raise Exception('Please enter the correct bert type.')
        self.drop_out = nn.Dropout(p=drop_out)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_class)

    def forward(self, batch, **kwargs):
        input_ids, attention_mask, token_type = batch[:3]
        try:
            out = self.bert(token_type_ids=token_type, attention_mask=attention_mask, \
                inputs_embeds=kwargs.get('embed'))[1]
        except:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type)[1]
            # embed =  self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type)
        out = self.drop_out(out)
        out = self.classifier(out)
        embed = None

        return out, embed

class RandomBert(nn.Module):
    def __init__(self, config, num_class):
        super(RandomBert, self).__init__()
        self.bert = BertModel(BertConfig())

        self.drop_out = nn.Dropout(p=config['hidden_dropout_prob'])
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_class)

    def forward(self, input_ids, attention_mask, token_type, **kwargs):
        try:
            out = self.bert(token_type_ids=token_type, attention_mask=attention_mask, \
                inputs_embeds=kwargs.get('input_embeds'))[1]
        except:
            if kwargs.get('return_embed') == None:
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type)[1]
            else:
                return self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type)
        out = self.drop_out(out)
        out = self.classifier(out)

        return out

class PretrainBert(nn.Module):
    def __init__(self, mask_token_id, mask_prob, place_prob):
        super(PretrainBert, self).__init__()

        # base setting about mask tokens
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        bert_config = BertConfig()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.drop_out = nn.Dropout(p=bert_config.hidden_dropout_prob)
        self.mlm_classifier = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)
    
    def forward(self, batch, **kwargs):
        # random mask some token to the sentence
        input_ids, attention_mask, token_type = batch[:3]

        try:
            out_mlm, out_nsp = self.bert(token_type_ids=token_type, attention_mask=attention_mask, \
                inputs_embeds=kwargs.get('input_embeds'))
        except:
            if kwargs.get('return_embed') == None:
                out_mlm, out_nsp = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type)
        out_mlm, out_nsp = self.drop_out(out_mlm), self.drop_out(out_nsp)
        out_mlm, out_nsp = self.mlm_classifier(out_mlm), self.nsp_classifier(out_nsp)

        return out, embed
    
    def get_loss(self, out, label, task):
        num_class = out['mlm'].shape[-1]
        if task == 'mlm':
            return F.cross_entropy(out['mlm'].reshape(-1, num_class), label['mlm'].reshape(-1), ignore_index=-1)
        elif task == 'nsp':
            return F.cross_entropy(out['nsp'], label['nsp'])
        else:
            print('TASK Error IN GET LOSS!')

class RandomPretrainBert(nn.Module):
    def __init__(self):
        super(RandomPretrainBert, self).__init__()

        bert_config = BertConfig()
        self.bert = BertModel(bert_config)

        self.drop_out = nn.Dropout(bert_config.hidden_dropout_prob)
        self.mlm_classifier = nn.Linear(bert_config.hidden_size, bert_config.vocab_size)
        self.nsp_classifier = nn.Linear(bert_config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask, token_type, **kwargs):
        try:
            out_mlm, out_nsp = self.bert(token_type_ids=token_type, attention_mask=attention_mask, \
                inputs_embeds=kwargs.get('input_embeds'))
        except:
            if kwargs.get('return_embed') == None:
                out_mlm, out_nsp = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type)
            else:
                return self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type)
        out_mlm, out_nsp = self.drop_out(out_mlm), self.drop_out(out_nsp)
        out_mlm, out_nsp = self.mlm_classifier(out_mlm), self.nsp_classifier(out_nsp)

        return {'mlm': out_mlm, 'nsp': out_nsp}
    
    def get_loss(self, out, label, task):
        num_class = out['mlm'].shape[-1]
        if task == 'mlm':
            return F.cross_entropy(out['mlm'].reshape(-1, num_class), label['mlm'].reshape(-1), ignore_index=-1)
        elif task == 'nsp':
            return F.cross_entropy(out['nsp'], label['nsp'])
        else:
            print('TASK Error IN GET LOSS!')
