import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, d_embed, d_hidden, n_vocab, p_drop):
        super(BiLSTMModel, self).__init__()
        self.Embedding = nn.Embedding(n_vocab, d_embed, padding_idx=0)
        self.lstm1 = nn.LSTM(d_embed, d_hidden // 2, dropout=p_drop, num_layers=2, batch_first=True, bidirectional=True)        
        self.lstm2 = nn.LSTM(d_hidden, d_hidden // 2, dropout=p_drop, batch_first=True, bidirectional=True)        
        # self.output_1 = nn.Linear(d_hidden * 2, int(d_hidden / 2))
        # self.output_2 = nn.Linear(int(d_hidden / 2), 3)
        self.out = nn.Linear(d_hidden, 3)
        self.drop_out = p_drop
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, batch, embed=None):
        x = batch[0]
        x_mask = batch[1]
        if embed == None:
            x_embed = self.Embedding(x)
        else:
            x_embed = embed

        # followings are rnn part
        original_length = x_mask.shape[1]
        lengths = torch.sum(x_mask, dim=-1)
        sort_length, sort_idx = torch.sort(lengths, descending=True)
        x_pack = nn.utils.rnn.pack_padded_sequence(x_embed[sort_idx], \
            sort_length, batch_first=True)
        output, _ = self.lstm1(x_pack)
        output, _ = self.lstm2(output)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=original_length)
        _, unsorted_idx = torch.sort(sort_idx)
        output = output[unsorted_idx]

        # outputlayer
        output = output[:, 0, :]
        output = self.out(output)

        return output, x_embed
    
    def fix_embedding(self, optimizer_state):
        pass
