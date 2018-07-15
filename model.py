'''
Author: MofaCatch 2018
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import functions.ctc as ctc
import torch.nn.functional as F
from ctc_decoder import decode

class Model(nn.Module):

    def __init__(self, input_dim, num_class, CUDA=False):
        super(Model, self).__init__()

        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=256,
                          num_layers=4,
                          batch_first=True,
                          dropout=0.6)

        self.fc = nn.Linear(256, num_class+1)
        self.optimizer = optim.Adam(self.parameters())
        self.cuda = CUDA
        self.ctc_loss = ctc.CTCLoss()

    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        return self.forward_imp(x), y, x_lens, y_lens

    def forward_imp(self, x, softmax=False):
        if self.cuda:
            x = x.cuda()
        x, _ = self.rnn(x)
        x = self.fc(x)

        if softmax: # for inference
            return F.softmax(x, dim=2)
        return x

    def train_model(self, batch):
        self.train()

        out, y, x_lens, y_lens = self.forward(batch)

        loss = self.ctc_loss(out, y, x_lens, y_lens)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_model(self, batch):
        self.eval()

        out, y, x_lens, y_lens = self.forward(batch)

        loss = self.ctc_loss(out, y, x_lens, y_lens)

        return loss.item()


    def collate(self, inputs, labels):
        max_t = max(i.shape[0] for i in inputs)
        #max_t = self.conv_out_size(max_t, 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        x = torch.FloatTensor(zero_pad_concat(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        batch = [Variable(v) for v in batch]

        return batch

    def infer(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        probs = self.forward_imp(x, softmax=True)
        probs = probs.cpu().detach().numpy()
        res = [decode(p, beam_size=1, blank=self.blank)[0] for p in probs]
        return res




    # def conv_out_size(self, n, dim):
    #     for c in self.conv.children():
    #         if type(c) == nn.Conv2d:
    #             # assuming a valid convolution
    #             k = c.kernel_size[dim]
    #             s = c.stride[dim]
    #             n = (n - k + 1) / s
    #             n = int(math.ceil(n))
    #     return n



def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t, inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0], :] = inp
    return input_mat