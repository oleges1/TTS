import torch.utils.data as data
import torchaudio
import numpy as np
import os


LEN_TRAIN_AUDIO = 500000


class ElectroDataset(data.Dataset):
    def __init__(self, root, subfolder='train', transforms=lambda x: x):
        self.dir = root
        self.subfolder = subfolder
        filenames = os.listdir(self.dir)
        num_train_files = int(len(filenames) * 0.95)
        self.train = filenames[:num_train_files]
        self.test = filenames[num_train_files:]
        self.transforms = transforms

    def __getitem__(self, idx):
        if self.subfolder == 'train':
            filename = self.train[idx]
        else:
            filename = self.test[idx]
        wav, sr = torchaudio.load(os.path.join(self.dir, filename))
        wav = wav[0]
        pos = np.random.choice(wav.shape[0] - LEN_TRAIN_AUDIO)
        return self.transforms({'audio' : wav[pos:pos + LEN_TRAIN_AUDIO], 'sample_rate': sr})

    def __len__(self):
        if self.subfolder == 'train':
            return len(self.train)
        return len(self.test)

class Electro2Dataset(data.Dataset):
    def __init__(self, root, subfolder='train', transforms=lambda x: x):
        self.dir = root
        self.subfolder = subfolder
        filenames = os.listdir(self.dir)
        num_train_files = int(len(filenames) * 0.9)
        self.train = filenames[:num_train_files]
        self.test = filenames[num_train_files:]
        self.transforms = transforms

    def __getitem__(self, idx):
        if self.subfolder == 'train':
            filename = self.train[idx]
        else:
            filename = self.test[idx]
        wav, sr = torchaudio.load(os.path.join(self.dir, filename))
        # wav = wav[0]
        # pos = np.random.choice(wav.shape[0] - LEN_TRAIN_AUDIO)
        return self.transforms({'audio' : wav[0], 'sample_rate': sr})

    def __len__(self):
        if self.subfolder == 'train':
            return len(self.train)
        return len(self.test)
