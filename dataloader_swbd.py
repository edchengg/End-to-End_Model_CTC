import numpy as np
import random
import torch.utils.data as tud
import json

class AudioDataset(tud.Dataset):

    def __init__(self, data_json, subset=0, log=True):

        data = read_data_json(data_json)

        if subset:
            data = data[:subset]

        bucket_diff = 4
        max_len = max(len(x['label']) for x in data)
        num_buckets = max_len // bucket_diff
        buckets = [[] for _ in range(num_buckets)]
        for d in data:
            bid = min(len(d['label']) // bucket_diff, num_buckets - 1)
            buckets[bid].append(d)

        # Sort by input length followed by output length
        sort_fn = lambda x: (x['duration'],
                             len(x['label']))
        for b in buckets:
            b.sort(key=sort_fn)
        data = [d for b in buckets for d in b]

        if log:
            for datum in data:
                new_audio = datum["audio"].replace('swbd_data', 'swbd_logfbankeddnorm')
                datum["audio"] = new_audio

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
 
        datum = self.data[idx]
        data = np.load(datum['audio'])
        datum = (data, datum["label"])
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
                batch_size, subset=0, num_workers=4, log=True):
    dataset = AudioDataset(dataset_json, subset=subset, log=log)
    sampler = BatchRandomSampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=lambda batch : zip(*batch),
                drop_last=True)
    return loader
