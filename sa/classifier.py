import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def eval_batch(model, input, inp_pos, inp_char, input_lengths):
    bsz = input.size(0)
    #print(inp.size())
    #model = model.cuda()
    with torch.no_grad():
        output = model(input, inp_pos, inp_char, input_lengths)
    # bsz
    output = F.softmax(output, dim=1)
    return output[:, 1]

def sort_by_lens(seq, seq_lengths):
    seq_lengths_sorted, sort_order = seq_lengths.sort(descending=True)
    seq_sorted = seq.index_select(0, sort_order)
    return seq_sorted, seq_lengths_sorted, sort_order


def unsort(x_sorted, sorted_order, dim=0):
    x_unsort = torch.zeros_like(x_sorted)
    if dim == 0:
        x_unsort[sorted_order] = x_sorted
    elif dim == 1:
        x_unsort[:, sorted_order] = x_sorted
    else:
        raise ValueError
    return x_unsort


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, emb_matrix, pad_idx, ncls, ninp, nhid, npos, dropout=0.5, tie_weights=False):
        """
        :param rnn_type:
        :param emb_matrix:
        :param pad_idx:
        :param ncls:
        :param ninp: emb size (dim)
        :param nhid: hidden dim
        :param npos: n of pos vocab
        :param dropout:
        :param tie_weights:
        """
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
        self.pos_embedding = nn.Embedding(npos, 50)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp+50+1, nhid, batch_first=True, dropout=dropout, bidirectional=True) # n: dim
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp+50+1, nhid, nonlinearity=nonlinearity, batch_first=True, dropout=dropout)
        self.decoder = nn.Sequential(nn.Linear(nhid * 2, 8), nn.Tanh(), nn.Linear(8, ncls), nn.Tanh())

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid

        self.dict = None # useful later

        self.pos_dict = None

    # def init_weights(self, emb_matrix):
    def init_weights(self):
        def init_sequential(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight.data)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
        # initrange = 0.1
        # self.encoder.weight.data.copy_(emb_matrix)
        # torch.nn.init.xavier_uniform(self.rnn.weight)
        # torch.nn.init.xavier_uniform(self.decoder.weight)
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.apply(init_sequential)

    def forward(self, input, input_pos, char_lens, input_lengths):
        emb = self.drop(self.embedding(input))
        pos_emb = self.drop(self.pos_embedding(input_pos))
        char_lens = char_lens.unsqueeze(-1)
        emb = torch.cat([emb, pos_emb, char_lens], dim=-1)
        emb_sorted, lengths_sorted, sort_order = sort_by_lens(emb, input_lengths)
        hidden = self.init_hidden(input.size(0))
        output, hidden = self.rnn(emb_sorted, hidden)
        #output = output.transpose(0,1)
        hidden = hidden[0].transpose(0, 1).contiguous().view(input.size(0), -1)
        decoded = self.decoder(hidden)#.view(output.size(0)*output.size(1), output.size(2)))
        decoded = unsort(decoded, sort_order, dim=0)
        return decoded

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(2, bsz, self.nhid),
                    weight.new_zeros(2, bsz, self.nhid))
        else:
            return weight.new_zeros(2, bsz, self.nhid)

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)
