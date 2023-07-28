import torch
from torch.utils.data import Dataset as DS
import Model.config as config

context_dimension = 10  # 512  # Time
embedding_dimension = 768  # 192#384  # feature Channels (Should be multiple of num_heads)
batch_size = 16  # 64  # Batch
num_heads = 12  # for multi-head Attention
num_blocks = 12
vocab_size = 50257
dropout = 0.2
device = 'cpu'

''' defines the amount of Samples per Sequence of size context_length
(1 corresponds to no overlap between samples, 2 results in approximately half-context overlap)'''
overlap = 2
# dataset = 'Bookcorpus/bc1.pt'
dataset_path = 'shakespeare_tokenized.pt'


class Dataset(DS):
    def __init__(self, cfg):
        self.context_length = cfg.context_length
        self.stepSize = cfg.context_length // overlap
        self.data = torch.load(dataset_path, map_location=torch.device('cpu'))
        print(self.data.size()[0])
        self.length = -(self.data.size()[0] // -self.stepSize)  # double minus to perform efficient ceil division

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        index = min(i * self.stepSize, self.data.size()[0]-self.context_length-1)
        x, y = self.data[index:index + self.context_length], self.data[index + 1:index + self.context_length + 1]
        return x, y


# '''
parameters = config.params(embedding_dimension, n_heads=num_heads, n_blocks=num_blocks,
                           batchsize=batch_size, context_length=context_dimension, vocab_size=vocab_size,
                           device=device, dropout=dropout)
ds = Dataset(parameters)
print(ds.__len__())
x, y = ds.__getitem__(67603)  # komisches Verhalten, wenn man um 1 reduziert
print(x)
print(y)
# '''
