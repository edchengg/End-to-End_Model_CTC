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

        self.rnn = getattr(nn,CONFIG['model']['model_type'])(input_size=input_dim,
                          hidden_size=CONFIG['model']['hid_dim'],
                          num_layers=CONFIG['model']['layer'],
                          batch_first=True,
                          dropout=CONFIG['model']['dropout'],
                          bidirectional=CONFIG['model']['bi'])

        self.bidirectional = CONFIG['model']['bi']
        self.fc = nn.Linear(CONFIG['model']['hid_dim'], num_class+1)

        self.use_cuda = CONFIG['cuda']
        self.ctc_loss = ctc.CTCLoss()
        self.blank = num_class
        self.beam_size = CONFIG['optimizer']['beam']

    def forward(self, inputs, labels):
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        return self.forward_imp(x), y, x_lens, y_lens

    def forward_imp(self, x, softmax=False):
        # Multi GPU setting does not require .cuda()
        if self.use_cuda:
            x = x.cuda()

        x, _ = self.rnn(x)

        if self.bidirectional:
            half = x.size(-1) // 2
            x = x[:, :, :half] + x[:, :, half:]

        x = self.fc(x)

        if softmax: # for inference
            return x, F.softmax(x, dim=2)
        return x

    def train_model(self, inputs, labels, optimizer):
        self.train()

        out, y, x_lens, y_lens = self.forward(inputs, labels)
        loss = self.ctc_loss(out, y, x_lens, y_lens)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval_model(self, inputs, labels):
        # Calculate loss and CER together
        self.eval()
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        out, probs = self.forward_imp(x, softmax=True)
        loss = self.ctc_loss(out, y, x_lens, y_lens)
        probs = probs.cpu().detach().numpy()
        pred = [decode(p, beam_size=self.beam_size, blank=self.blank)[0] for p in probs]
        return loss.item(), pred


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

    def infer(self, inputs, labels):
        x, y, x_lens, y_lens = self.collate(inputs, labels)
        probs = self.forward_imp(x, softmax=True)
        probs = probs.cpu().detach().numpy()
        res = [decode(p, beam_size=self.beam_size, blank=self.blank)[0] for p in probs]
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
