import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import functions.ctc as ctc
import torch.nn.functional as F
from ctc_decoder import decode

class Model(nn.Module):

    def __init__(self, input_dim, num_class, CONFIG):
        super(Model, self).__init__()

        self.bidirectional = True
        self._hid_dim = 256
        self._rnn2_input_size = self._hid_dim * 2 if self.bidirectional else self._hid_dim
        self._ffn_input_size = self._hid_dim * 2 if self.bidirectional else self._hid_dim

        # Define RNN model
        self.rnn1 = nn.LSTM(input_size=input_dim,
                            hidden_size=256,
                            num_layers=1,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)

        self.rnn2 = nn.LSTM(input_size=self._rnn2_input_size,
                            hidden_size=256,
                            num_layers=1,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)

        self.rnn3 = nn.LSTM(input_size=self._rnn2_input_size,
                            hidden_size=256,
                            num_layers=1,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)

        # Define FFN after RNN
        self.fc = nn.Linear(self._ffn_input_size, num_class + 1)

        self.dropout = nn.Dropout(0.1)

        self.use_cuda = CONFIG['cuda']
        self.ctc_loss = ctc.CTCLoss()
        self.blank = num_class
        self.clip = CONFIG['optimizer']['clip']
        self.reduce_factor = 1

    def forward(self, inputs, labels):
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        return self.forward_imp(x), y, x_lens, y_lens

    def forward_imp(self, x, softmax=False):
        if self.use_cuda:
            x = x.cuda()

        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)
        x, _ = self.rnn3(x)
        x = self.dropout(x)
        # Get probability distribution across all time steps
        # Output x is a n_vocab distribution
        x = self.fc(x)

        # Use softmax to get distribution
        if softmax: # for inference
            return x, F.softmax(x, dim=2)
        return x

    def train_model(self, inputs, labels, optimizer):
        self.train()

        # Get output distribution, label, input length, label length
        out, y, x_lens, y_lens = self.forward(inputs, labels)
        # Calculate CTC loss
        loss = self.ctc_loss(out, y, x_lens, y_lens)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        optimizer.step()

        return loss.item()

    def eval_model(self, inputs, labels, beam_size=1):
        # Calculate loss and CER together
        self.eval()
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        out, probs = self.forward_imp(x, softmax=True)
        loss = self.ctc_loss(out, y, x_lens, y_lens)
        # Turn to cpu
        probs = probs.cpu().detach().numpy()
        # Use CTC decoder beam size to get prediction of characters
        x_lens = x_lens.cpu().detach().numpy()
        if beam_size==1:
            pred = [max_decode(p[:l], blank=self.blank) for p, l in zip(probs, x_lens)]
        else:
            pred = [decode(p[:l], beam_size=beam_size, blank=self.blank)[0] for p, l in zip(probs, x_lens)]
        return loss.item(), pred

    
    def collate(self, inputs, labels):
        # Record input length
        lens = [len(i) for i in inputs]
        final_lens = []
        for i in lens:
            if i % self.reduce_factor != 0:
                i += self.reduce_factor - (i % self.reduce_factor)
            j = i // self.reduce_factor
            final_lens.append(j)

        x_lens = torch.IntTensor(final_lens)
        x = torch.FloatTensor(zero_pad_concat(inputs, self.reduce_factor))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        batch = [Variable(v) for v in batch]

        return batch

    def infer(self, inputs, labels):
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        out, probs = self.forward_imp(x, softmax=True)
        probs = probs.cpu().detach().numpy()  
        x_lens = x_lens.cpu().detach().numpy()
        res = [decode(p[:l], beam_size=self.beam_size, blank=self.blank)[0] for p,l in zip(probs, x_lens)]
        return res


def max_decode(pred, blank):
    prev = np.argmax(pred[0])
    seq = [prev] if prev != blank else []
    for prob in pred[1:]:
        p = np.argmax(prob)
        if p != blank and p != prev:
            seq.append(p)
        prev = p
    return seq



def zero_pad_concat(inputs, reduce_factor):
    max_t = max(inp.shape[0] for inp in inputs)
    if max_t % reduce_factor != 0:
        max_t += reduce_factor - (max_t%reduce_factor)

    shape = (len(inputs), max_t, inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0], :] = inp
    return input_mat