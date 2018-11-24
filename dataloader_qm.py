import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.utils.data as tud
import json

class AudioDataset(tud.Dataset):

    def __init__(self, data_json, subset=0, norm=True):

        data = read_data_json(data_json)

        if subset:
            data = data[:subset]

        bucket_diff = 4
        max_len = max(len(x['text']) for x in data)
        num_buckets = max_len // bucket_diff
        buckets = [[] for _ in range(num_buckets)]
        for d in data:
            bid = min(len(d['text']) // bucket_diff, num_buckets - 1)
            buckets[bid].append(d)

        # Sort by input length followed by output length
        sort_fn = lambda x: (x['length'],
                             len(x['text']))
        for b in buckets:
            b.sort(key=sort_fn)
        data = [d for b in buckets for d in b]

        self.data = data

        # use normalized feature
        if norm:
            for datum in self.data:
                new_audio = datum["audio"].replace('qm', 'norm')
                datum["audio"] = new_audio
        else:
            for datum in self.data:
                new_audio = datum["audio"].replace('qm', 'unnorm')
                datum["audio"] = new_audio

        print(self.data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
 
        datum = self.data[idx]
        datum = (np.load(datum["audio"]), datum["text"])
        return datum


def read_data_json(data_json):
    with open(data_json) as fid:
        return [json.loads(l) for l in fid]

class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size)
                for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)

def make_loader(dataset_json,
                batch_size, subset=0, num_workers=4, norm=False):
    dataset = AudioDataset(dataset_json,
                           subset, norm)
    sampler = BatchRandomSampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=lambda batch : zip(*batch),
                drop_last=True)
    return loader
