import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, AlbertModel, BertConfig

class Bert(nn.Module):
    def __init__(self, bert_name, num_class, drop_out=0.1, sequence_label=False):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.sequence_label = sequence_label
        self.drop_out = nn.Dropout(p=drop_out)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_class)

    def forward(self, batch, **kwargs):
        input_ids, attention_mask, token_type = batch[:3]
        try:
            sequence_out, pooler_out = self.bert(token_type_ids=token_type, attention_mask=attention_mask, \
                inputs_embeds=kwargs.get('embed'))
        except:
            sequence_out, pooler_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type)
        out = sequence_out if self.sequence_label else pooler_out
        out = self.drop_out(out)
        out = self.classifier(out)
        embed =  self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type)

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
