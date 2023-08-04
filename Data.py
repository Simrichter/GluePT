import torch
from torch.utils.data import Dataset as DS

overlap = 2
dataset_path = 'Bookcorpus/bc1.pt'
#dataset_path = 'shakespeare_tokenized.pt'


class Dataset(DS):
    def __init__(self, cfg, train, train_test_percentage):
        self.context_length = cfg.context_length
        self.stepSize = cfg.context_length // overlap
        self.data = torch.load(dataset_path, map_location=torch.device('cpu'))

        #calculate indices for train-test split (create a ~uniform test distribution)
        num_samples = int(train_test_percentage * len(self.data))
        stride = len(self.data) // num_samples
        test_indices = [i * stride for i in range(num_samples)]

        #either select or delete sequences starting from calculated indices
        '''if train:
            mask = torch.ones(self.data.numel(), dtype=torch.bool)
            for i in tqdm(test_indices):
                mask[i:i + self.context_length] = False
            self.data = self.data[mask]
            torch.save(self.data, 'train_set.pt')
        else: # test
            mask = torch.zeros(self.data.numel(), dtype=torch.bool)
            for i in tqdm(test_indices):
                mask[i:i + self.context_length] = True
            self.data = self.data[mask]
            #temp = torch.empty(0)
            #for index in tqdm(test_indices):
            #    temp = torch.cat((temp, self.data[index:index + self.context_length]))
            #self.data = temp
            torch.save(self.data, 'test_set.pt')'''

        self.data = self.data[:int(train_test_percentage * len(self.data))] if train else self.data[int(train_test_percentage * len(self.data)):]
        self.length = -((self.data.size()[0]-1) // -self.stepSize)  # double minus to perform efficient ceil division

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        assert self.length > i >= 0
        index = min(i * self.stepSize, self.data.size()[0]-self.context_length-1)
        x, y = self.data[index:index + self.context_length], self.data[index + 1:index + self.context_length + 1]
        return x, y
