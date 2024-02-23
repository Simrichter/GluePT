import torch
from torch.utils.data import Dataset
import Model.Embeddings as emb
from tqdm import tqdm
from datasets import load_dataset
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
    "ax": ("premise", "hypothesis")
}

# This class was used for testing, as it implements a custom dataset
class ExternalDataset(Dataset):
    def __init__(self, data, x='x', y='y'):
        self.data = data
        self.length = len(data)
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length

# This class represents one of the GLUE datasets
class FinetuneData(Dataset):
    def __init__(self, task, split):
        data_path = f'Glue/glue_{task}.pt'

        # Downloads and saves the necessary dataset if not already done
        if not os.path.exists(data_path):
            print(f"downloading {task}")
            prepare_glue([task])
        self.data = torch.load(data_path)[split] # Loads the correct split of the dataset (train/test/validation)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        assert 0 <= i < len(self)
        return self.data[i]

# This class represents the OpenWebText dataset
# It manages the individual subset files and the creation of samples and labels
class Dataset(Dataset):

    def __init__(self, T, train, num_subsets, train_test_percentage=0.995, overlap = 2):
        # Overlap defines the amount of Samples per Sequence of size context_length
        # (1 corresponds to no overlap between samples,
        # 2 will result in approximately half-context overlap,
        # set to context_dimension for full overlap)'''
        
        torch.multiprocessing.set_sharing_strategy('file_system') # Solves an error with multiple workers ("Bad file descriptor")

        self.context_length = T
        self.stepSize = T // overlap

        dataset_paths = ['OpenWebText/owt_{}.pt'.format(i) for i in range(num_subsets)]
        
        if train:
            print(f"Using {num_subsets} OpenWebText subset{'s' if num_subsets>1 else ''}")

        prepare_owt(end=num_subsets) # Download owt files if not already done

        self.tensors = [torch.load(path, map_location=torch.device('cpu')) for path in dataset_paths]

        # splits the test part of the end of each individual file and concatenates the resulting tensor
        self.data = torch.cat( [data[:int(train_test_percentage * len(data))] if train else data[int(train_test_percentage * len(data)):] for data in self.tensors] )

        self.length = -((self.data.size()[0] - 1) // -self.stepSize) # double minus to perform ceil division

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i >= self.length:
            print(f"i:{i}, len:{self.length}")
        assert 0 <= i < self.length
        index = min(i * self.stepSize, self.data.size()[0] - self.context_length - 1) # The start index of the sample in the dataset
        x, y = self.data[index: (index + self.context_length)], self.data[index + 1:index + self.context_length + 1] # load sample and label
        return x, y
    
# This class allows to resume pretraining in an epoch by saving the order, in which the samples are trained
class ResumableSampler(torch.utils.data.Sampler):
    def __init__(self, length):
        self.offset = 0
        self.length = length-self.offset

        self.perm_index = -1
        self.perm = torch.randperm(self.length, pin_memory=torch.cuda.is_available(), device='cpu')

    def __iter__(self):
        # yield the next sample index
        while self.perm_index < len(self.perm)-1:
            self.perm_index += 1
            yield self.perm[self.perm_index]
        
        # If all indices have been yielded, the random permutation is reset
        self.length += self.offset
        self.offset = 0
        self.perm_index = -1
        self.perm = torch.randperm(self.length, pin_memory=torch.cuda.is_available(), device='cpu')

    def __len__(self):
        return self.length
    
    # Returns a state to be saved together with th model checkpoint
    def get_state(self):
        return {"perm": self.perm}

    # Sets the state when loading from a checkpoint
    def set_state(self, state, i):
        self.perm = state["perm"]
        self.perm_index = i-1
        self.offset = i
        self.length = self.length - self.offset

# This function downloads the OpenWebText subsets
# It only accesses the auto-converted parquet files of Huggingface and can therefore only load up to the first 5GB of OpenWebText
def prepare_owt(start=0, end=10):
    global task_to_keys
    
    subset_ids = [i for i in range(start, end)]

    for id in subset_ids:
        if not os.path.exists(f'OpenWebText/owt_{id}.pt'):
            print(f"Downloading OpenWebText subset {id}")
            url = 'https://huggingface.co/datasets/Skylion007/openwebtext/resolve/refs%2Fconvert%2Fparquet/plain_text/partial-train/00{}.parquet'.format(str(id).zfill(2))
            data = load_dataset("parquet", data_files={"train": url}, split="train")
            res = []
            for s in data['text']:
                res.append(torch.tensor(emb.encode(s)))
            if not os.path.exists("OpenWebText"):
                os.mkdir("OpenWebText")
            torch.save(torch.cat(res), 'OpenWebText/owt_{}.pt'.format(id))

# This function downloads the GLUE datasets and saves them for later use
def prepare_glue(tasks):
    if not os.path.exists("Glue"):
        os.mkdir("Glue")    

    for task in tasks:
        if task == 'mnli':
            splits = ['train', 'test_matched', 'test_mismatched', 'validation_matched', 'validation_mismatched']
        elif task == 'ax':
            splits = ['test']
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
            dataset[split]=res
        torch.save(dataset, 'Glue/glue_{}.pt'.format(task))
