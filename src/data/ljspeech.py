from torch.utils import data
import torchaudio
import os
from torch.utils.data import Subset

class LJSpeechDataset(torchaudio.datasets.LJSPEECH):
    def __init__(self, transforms=lambda x: x, keys=['audio', 'text', 'sample_rate'], *args, **kwargs):
        if kwargs.get('download', False):
            os.makedirs(kwargs['root'], exist_ok=True)
        super(LJSpeechDataset, self).__init__(*args, **kwargs)
        self.transforms = transforms
        self.keys = keys

    def __getitem__(self, idx):
        audio, sample_rate, _, norm_text = super().__getitem__(idx)
        res_dict = {}
        for key in self.keys:
            if key == 'audio':
                res_dict[key] = audio
            elif key == 'text':
                res_dict[key] = norm_text
            elif key == 'sample_rate':
                res_dict[key] = sample_rate
        return self.transforms(res_dict)

    def get_text(self, n):
        line = self._walker[n]
        fileid, transcript, normalized_transcript = line
        return self.transforms({'text' : normalized_transcript})['text']



def get_dataset(config, transforms=lambda x: x, part='train', download=False, *args, **kwargs):
    if part == 'train':
        dataset = LJSpeechDataset(root=config.dataset.root, download=download, transforms=transforms, *args, **kwargs)
        indices = list(range(len(dataset)))
        dataset = Subset(dataset, indices[:int(config.dataset.get('train_part', 0.95) * len(dataset))])
        return dataset
    elif part == 'val':
        dataset = LJSpeechDataset(root=config.dataset.root, download=download, transforms=transforms, *args, **kwargs)
        indices = list(range(len(dataset)))
        dataset = Subset(dataset, indices[int(config.dataset.get('train_part', 0.95) * len(dataset)):])
        return dataset
    else:
        raise ValueError('Unknown')
