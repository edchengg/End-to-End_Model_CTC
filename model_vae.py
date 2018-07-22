import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from ctc_decoder import decode

class VAE(nn.Module):

    def __init__(self, input_dim, CONFIG):
        super(VAE, self).__init__()

        self.bi_encoder = CONFIG['encoder']['bi']
        self.bi_decoder = CONFIG['decoder']['bi']

        self.encoder = getattr(nn,CONFIG['encoder']['model_type'])(input_size=input_dim,
                          hidden_size=CONFIG['encoder']['hid_dim'],
                          num_layers=CONFIG['encoder']['layer'],
                          batch_first=True,
                          dropout=CONFIG['encoder']['dropout'],
                          bidirectional=CONFIG['encoder']['bi'])

        if self.bi_encoder:
            self.gauss_mean = nn.Linear(2 * CONFIG['encoder']['layer'] * CONFIG['encoder']['hid_dim'],
                                        CONFIG['model']['latent_dim'])
            self.gauss_log_var = nn.Linear(2 * CONFIG['encoder']['layer'] * CONFIG['encoder']['hid_dim'],
                                        CONFIG['model']['latent_dim'])
        else:
            self.gauss_mean = nn.Linear(CONFIG['encoder']['layer'] * CONFIG['encoder']['hid_dim'],
                                        CONFIG['model']['latent_dim'])
            self.gauss_log_var = nn.Linear(CONFIG['encoder']['layer'] * CONFIG['encoder']['hid_dim'],
                                        CONFIG['model']['latent_dim'])

        self.decoder = getattr(nn,CONFIG['decoder']['model_type'])(input_size=CONFIG['model']['latent_dim'],
                          hidden_size=CONFIG['decoder']['hid_dim'],
                          num_layers=CONFIG['decoder']['layer'],
                          batch_first=True,
                          dropout=CONFIG['decoder']['dropout'],
                          bidirectional=CONFIG['decoder']['bi'])

        self.fc = nn.Linear(CONFIG['decoder']['hid_dim'], input_dim)

        self.use_cuda = CONFIG['cuda']
        self.encoder_is_lstm = True if CONFIG['encoder']['model_type'] == 'LSTM' else False
        self.decoder_is_lstm = True if CONFIG['decoder']['model_type'] == 'LSTM' else False
        self.loss = nn.MSELoss(size_average=False)

    def forward(self, inputs):
        _, seq_len, _ = inputs.size()
        mean, log_var = self.encoding(inputs)
        z = self.reparameterize(mean, log_var)

        # cat z seq len times
        z_sequence = z.repeat(1, seq_len, 1)

        output = self.decoding(z_sequence)
        return output, mean, log_var

    def encoding(self, x):
        if self.encoder_is_lstm:
            # LSTM output: (seq hids ,(hid, cell))
            _, (hid, _) = self.encoder(x)
        else:
            # GRU output: (seq hids, hid)
            _, hid = self.encoder(x)

        # (num_layer, batch, hid_size) -> (batch, num_layer, hid_size))
        hid = torch.transpose(hid, 0, 1)
        batch, _, _ = hid.size()

        hid = hid.contiguous().view(batch, 1, -1)
        mean = self.gauss_mean(hid)
        log_var = self.gauss_log_var(hid)
        return mean, log_var

    def decoding(self, x):
        if self.decoder_is_lstm:
            out, (_, _) = self.decoder(x)
        else:
            out, _ = self.decoder(x)

        if self.bi_decoder:
            half = out.size(-1) // 2
            out = out[:, :, :half] + out[:, :, half:]
        out = self.fc(out)
        return out

    def reparameterize(self, mean, log_var):
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if self.use_cuda:
                eps = eps.cuda()
            return eps.mul(std).add_(mean)
        else:
            return mean

    def train_model(self, inputs, optimizer, LAMBDA):
        self.train()
        inputs = torch.FloatTensor(zero_pad_concat(inputs))
        if self.use_cuda:
            inputs = inputs.cuda()
        out, mean, log_var = self.forward(inputs)
        recon_loss = self.loss(out, inputs)
        kl_divergence = torch.sum(0.5 * (mean ** 2 + torch.exp(log_var) - log_var - 1))
        loss = recon_loss + LAMBDA * kl_divergence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), recon_loss.item(), kl_divergence.item()

    def eval_model(self, inputs, LAMBDA):
        self.eval()
        inputs = torch.FloatTensor(zero_pad_concat(inputs))
        if self.use_cuda:
            inputs = inputs.cuda()
        out, mean, log_var = self.forward(inputs)
        recon_loss = self.loss(out, inputs)
        kl_divergence = torch.sum(0.5 * (mean ** 2 + torch.exp(log_var) - log_var - 1))
        loss = recon_loss + LAMBDA * kl_divergence
        return loss.item(), recon_loss.item(), kl_divergence.item()

def zero_pad_concat(inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape = (len(inputs), max_t, inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        input_mat[e, :inp.shape[0], :] = inp
    return input_mat


