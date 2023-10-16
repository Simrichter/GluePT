import torch
from torch.utils.data import Dataset
import Model.config as config
import Model.Embeddings as emb
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import torch.multiprocessing
import os

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class ExternalDataset(Dataset):
    def __init__(self, data, x='x', y='y'):
        self.data = data
        self.length = len(data)
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # print(self.data[index][self.x])
        return self.data[index]  # [self.x], self.data[index][self.y]

    def __len__(self):
        return self.length


class FinetuneData(Dataset):
    def __init__(self, task, split):
        data_path = f'Glue/glue_{task}.pt'
        if not os.path.exists(data_path):
            print(f"downloading {task}")
            prepare_glue([task])
        self.data = torch.load(data_path)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        assert 0 <= i < len(self)
        return self.data[i]  # [0], self.data[i][1]


class Dataset(Dataset):

    # dataset_path = 'shakespeare_tokenized.pt'
    def __init__(self, cfg, train, train_test_percentage):
        torch.multiprocessing.set_sharing_strategy(
            'file_system')  # Solves an error with multiple workers ("Bad file descriptor")

        ''' defines the amount of Samples per Sequence of size context_length
        (1 corresponds to no overlap between samples,
         2 will result in approximately half-context overlap,
         set to context_dimension for full overlap)'''
        overlap = 2
        num_subsets = 10

        dataset_paths = ['OpenWebText/owt_{}.pt'.format(i) for i in range(num_subsets)]  # 'Bookcorpus/bc1.pt'

        self.context_length = cfg.context_length
        self.stepSize = cfg.context_length // overlap

        print("loading {} {} subsets".format(num_subsets, 'training' if train else 'validation'))
        self.tensors = [torch.load(path, map_location=torch.device('cpu')) for path in tqdm(dataset_paths)]

        # splits the test part of the end of each individual file and concatenates the resulting tensor
        self.data = torch.cat(
            [data[:int(train_test_percentage * len(data))] if train else data[int(train_test_percentage * len(data)):]
             for data in self.tensors])

        self.length = -((self.data.size()[0] - 1) // -self.stepSize)  # double minus to perform efficient ceil division

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i >= self.length:
            print(f"i:{i}, len:{self.length}")
        assert 0 <= i < self.length
        index = min(i * self.stepSize, self.data.size()[0] - self.context_length - 1)
        # print(index)
        # print(self.data[0:1])
        x, y = self.data[index: (index + self.context_length)], self.data[index + 1:index + self.context_length + 1]
        # print(index)
        return x, y


class ResumableSampler(torch.utils.data.Sampler):
    def __init__(self, length):
        self.offset = 0
        self.length = length - self.offset

        self.perm_index = -1
        self.perm = torch.randperm(self.length, pin_memory=True, device='cpu')
        # self.perm = self.perm.to('cpu')

    def __iter__(self):
        while self.perm_index < len(self.perm) - 1:
            self.perm_index += 1
            # self.log.append(self.perm[self.perm_index])
            yield self.perm[self.perm_index]
        self.length += self.offset
        self.offset = 0
        self.perm_index = -1
        self.perm = torch.randperm(self.length, pin_memory=True, device='cpu')  # generator=self.generator

    def __len__(self):
        return self.length

    def get_state(self):
        return {"perm": self.perm}

    def set_state(self, state, i):
        self.perm = state["perm"]
        self.perm_index = i - 1
        self.offset = i  # state["perm_index"] if not self.perm_index >= len(self.perm) else -1
        self.length = self.length - self.offset


def prepare_owt():
    global task_to_keys

    subset_ids = [i for i in range(4, 10)]

    for id in tqdm(subset_ids):
        # url = 'https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset{}.tar'.format(str(id).zfill(2))
        url = 'https://huggingface.co/datasets/Skylion007/openwebtext/resolve/refs%2Fconvert%2Fparquet/plain_text/partial-train/00{}.parquet'.format(
            str(id).zfill(2))
        data = load_dataset("parquet", data_files={"train": url}, split="train")
        res = []
        for s in tqdm(data['text']):
            res.append(torch.tensor(emb.encode(s)))
        torch.save(torch.cat(res), 'OpenWebText/owt_{}.pt'.format(id))


def prepare_glue(tasks):
    for task in tasks:
        if task == 'mnli':
            splits = ['train', 'test_matched', 'test_mismatched', 'validation_matched', 'validation_mismatched']
        else:
            splits = ['train', 'test', 'validation']
        dataset = {}
        key_1, key_2 = task_to_keys[task]
        for split in splits:
            data = load_dataset('glue', task, split=split)
            res = []
            for sample in tqdm(data):
                x = torch.tensor(emb.encode(sample[key_1]))
                if key_2 is not None:
                    x = [x, torch.tensor(emb.encode(sample[key_2]))]
                res.append([x, torch.tensor([sample['label']])])
            dataset[split] = res
        torch.save(dataset, 'Glue/glue_{}.pt'.format(task))
# prepare_glue()
# prepare_owt()
