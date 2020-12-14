# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./Family_Relationships/cls/',
                    help='location of the data corpus')
parser.add_argument('--vocab_path', type=str, default='vocab_30k.txt')
parser.add_argument('--vocab_size', type=int, default=30000)
parser.add_argument('--glove_path', type=str, help='/path/to/glove/emb')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
# parser.add_argument('--bptt', type=int, default=40,
#                     help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default=None,
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

vocab = data.Vocab()
if not os.path.isfile(args.vocab_path):
    print('** no vocab file is found. now generate.')
    data_path = args.data + 'train.tok.csv'
    vocab.get_vocab_file(data_path, args.vocab_path, args.vocab_size)
vocab.add_vocab_from_file(args.vocab_path, args.vocab_size)
vocab.add_embedding(args.glove_path, args.emsize)

corpus = data.Corpus(args.data, vocab)


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def metric(output, target):
    pred = torch.argmax(output)[1]
    acc = torch.sum(pred.eq(target)) * 1. / len(target)
    pos = target.eq(1)
    neg = target.eq(0)
    tp = (pred.eq(1)).eq(pos).sum()
    fn = (pred.eq(0)).ne(neg).sum()
    tn = (pred.eq(0)).eq(neg).sum()
    fp = (pred.eq(1)).ne(neg).sum()
    tpr = tp * 1. / (tp + fn)
    tnr = tn * 1. / (tn + fp)
    return acc, tpr, tnr
# def batchify(data, bsz):
#     # Work out how cleanly we can divide the dataset into bsz parts.
#     # nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     # data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = torch.split(data, bsz, dim=0)
#     return data

def batchify(data, lens, tgt, bsz):
    data = torch.split(data, bsz, dim=0)
    lens = torch.split(lens, bsz, dim=0)
    tgt = torch.split(tgt, bsz, dim=0)
    return data, lens, tgt


eval_batch_size = 10
# train_data total sents: 532,249
train_data, train_lens, train_tgt = batchify(corpus.train_src, corpus.train_lens, corpus.train_tgt, args.batch_size)
val_data, val_lens, val_tgt = batchify(corpus.valid_src, corpus.valid_lens, corpus.valid_tgt, eval_batch_size)
test_data, test_lens, test_tgt = batchify(corpus.test_src, corpus.test_lens, corpus.test_tgt, eval_batch_size)

print(train_data[-1].size(), train_lens[-1].size(), train_tgt[-1].size())
###############################################################################
# Build the model
###############################################################################

ncls = 2
model = model.RNNModel(args.model, vocab.embedding, vocab.word2idx['<blank>'], ncls, args.emsize,
                       args.nhid, args.dropout, args.tied).to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=args.lr)


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, lens, tgt, i):
    # seq_len = min(args.bptt, len(source) - 1 - i)
    # data = source[i:i+seq_len]
    # target = source[i+1:i+1+seq_len].view(-1)
    data = source[i]
    len_ = lens[i]
    target = tgt[i]
    # target = source[i][1:]
    return data.cuda(), len_.cuda(), target.cuda()


def evaluate(src_data, src_lens, tgt_data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    acc_mean = 0.
    recall_mean = 0.
    prec_mean = 0.
    with torch.no_grad():
        for i in range(0, len(src_data)):
            data, lens, targets = get_batch(src_data, src_lens, tgt_data, i)
            output = model(data, lens)
            #hidden = repackage_hidden(hidden)
            acc, recall, prec = metric(output, targets)
            acc_mean += acc
            recall_mean += recall
            prec_mean += prec
    return acc_mean / (i+1), recall_mean / (i+1), prec_mean / (i+1)

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data))):
        data, lens, targets = get_batch(train_data, train_lens, train_tgt, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data, lens)
        acc, recall, prec = metric(output, targets)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)
        optim.step()
        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | acc {:5.2f} | recall {:5.2f} | prec {:5.2f}'.format(
                    epoch, batch, len(train_data), lr, elapsed * 1000 / args.log_interval, 
                    cur_loss, acc, recall, prec))
            total_loss = 0
            start_time = time.time()

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_acc = None

# At any point you can hit Ctrl + C to break out of training early.
if True:
    try:
        for epoch in range(1, args.epochs + 1):
            if epoch == 1:
                with open(os.path.join(args.save_dir, 'vocab.bin'), 'wb') as f:
                    torch.save(vocab.word2idx, f)

            epoch_start_time = time.time()
            train()
            acc, recall, prec = evaluate(val_data, val_lens, val_tgt)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid acc {:5.2f} | valid recall {:5.2f} | valid prec {:5.2f}'
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             acc, recall, prec))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_acc or acc < best_val_acc:
                with open(os.path.join(args.save_dir, 'model.pt'), 'wb') as f:
                    torch.save(model, f)
                best_val_acc = acc
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
else:
    val_loss = evaluate(val_data)
    print(val_loss)

# Load the best saved model.
with open(os.path.join(args.save_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=40)
