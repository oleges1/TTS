import os
import math
import torch
import wandb
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.ljspeech import get_dataset
from data.transforms import (
    MelSpectrogram, Compose, AddLengths, Pad,
    ToNumpy, AudioSqueeze, ToGpu, AudioEncode
)
from data.collate import no_pad_collate
from data.transforms import MelSpectrogramConfig
from wavenet.model.net import WaveNet
from utils import fix_seeds, mu_law_decode_torch

class WaveNetTrainer(pl.LightningModule):
    def __init__(
           self,
           config
        ):
        super(WaveNetTrainer, self).__init__()
        self.config = config
        fix_seeds(seed=config.train.seed)
        self.model = WaveNet(**config.model)
        self.lr = config.train.lr
        self.batch_size = config.train.batch_size
        self.weight_decay = config.train.get('weight_decay', 0.)
        self.num_workers = config.train.get('num_workers', 4)
        self.step_size = config.train.get('step_size', 15)
        self.gamma = config.train.get('optim_gamma', 0.5)
        self.quantization_channels = config.model.get('n_classes', 15)
        self.sample_rate = MelSpectrogramConfig.sr

        self.train_max_wav_len = config.train.get('train_max_wav_len', 20000)
        self.train_max_mel_len = math.ceil(self.train_max_wav_len / MelSpectrogramConfig.hop_length)
        self.validation_wav_len = config.train.get('validation_wav_len', 2048)
        self.validation_mel_len = math.ceil(self.validation_wav_len / MelSpectrogramConfig.hop_length)

        self.audio_transform = AudioEncode(quantization_channels=self.quantization_channels)
        self.mel = MelSpectrogram()
        self.gpu = ToGpu('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocess = Pad()
        self.crossentropy = nn.CrossEntropyLoss(ignore_index=-1) # ignore padding
        self.epoch_idx = 0

    def forward(self, batch):
        if self.training:
            # use 100% teacher forcing:
            hot_tensor = F.one_hot(batch['audio_quantized'][..., :-1].clamp(min=0), num_classes=self.quantization_channels)
            return self.model(
                x=hot_tensor.permute(0, 2, 1).float(),
                h=batch['mel'][..., 1:]
            )
        else:
            return self.model.generate(
                x=torch.empty((1, self.quantization_channels, 0), device=batch['audio_quantized'].device).float(),
                h=batch['mel'],
                samples=batch['audio_quantized'].shape[-1]
            )


    def training_step(self, batch, batch_nb):
        # REQUIRED
        batch = self.mel(self.gpu(batch))
        batch = self.preprocess(batch)
        batch['audio_quantized'] = batch['audio_quantized'][..., :self.train_max_wav_len]
        batch['mel'] = batch['mel'][..., :self.train_max_mel_len]
        logprobs = self(batch)
        classes = logprobs.argmax(dim=1)
        loss = self.crossentropy(logprobs, batch['audio_quantized'][..., 1:])
        losses_dict = {
            'train_loss': loss.item(),
            'train_acc': (batch['audio_quantized'][..., 1:] == classes).sum() / classes.shape[-1] / classes.shape[0]
        }
        self.logger.experiment.log(losses_dict)
        if batch_nb % self.config.train.train_log_period == 1:
            examples = [
                wandb.Image(batch['mel'][0].detach().cpu().numpy(), caption='mel'),
            ]
            self.logger.experiment.log({
                "plots_train": examples
            })
            examples = []
            predicted_audio = mu_law_decode_torch(classes[0])
            examples.append(wandb.Audio(predicted_audio.detach().cpu().numpy(), caption='reconstructed_wav', sample_rate=self.sample_rate))
            examples.append(wandb.Audio(batch['audio'][0].detach().cpu().numpy(), caption='target_wav', sample_rate=self.sample_rate))
            self.logger.experiment.log({
                "audios_train": examples
            })
        return loss


    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        batch = self.mel(self.gpu(batch))
        batch = self.preprocess(batch)
        batch['audio_quantized'] = batch['audio_quantized'][..., :self.validation_wav_len]
        batch['mel'] = batch['mel'][..., :self.validation_mel_len]
        predictions = self(batch)
        losses_dict = {
            'val_acc': (batch['audio_quantized'] == predictions).sum() / predictions.shape[-1]
        }
        self.logger.experiment.log(losses_dict)
        if batch_nb % self.config.train.train_log_period == 1:
            examples = [
                wandb.Image(batch['mel'][0].detach().cpu().numpy(), caption='mel'),
            ]
            self.logger.experiment.log({
                "plots_val": examples
            })
            examples = []
            predicted_audio = mu_law_decode_torch(predictions[0])
            examples.append(wandb.Audio(predicted_audio.detach().cpu().numpy(), caption='reconstructed_wav', sample_rate=self.sample_rate))
            examples.append(wandb.Audio(batch['audio'][0].detach().cpu().numpy(), caption='target_wav', sample_rate=self.sample_rate))
            self.logger.experiment.log({
                "audios_val": examples
            })
        return losses_dict

    def validation_epoch_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        keys = outputs[0].keys()
        logdict = {}
        for key in keys:
            logdict[f'avg_{key}'] = torch.stack([x[key] for x in outputs]).mean().item()
        self.logger.experiment.log(logdict)

        os.makedirs(self.config.train.get('checkpoint_path', 'checkpoints'), exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.train.get('checkpoint_path', 'checkpoints'), f'model_{self.epoch_idx}.pth')
        )
        self.logger.experiment.save(os.path.join(self.config.train.get('checkpoint_path', 'checkpoints'), f'model_{self.epoch_idx}.pth'))
        self.epoch_idx += 1

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

    # dataset:

    def prepare_data(self):
        get_dataset(self.config, download=True)

    def train_dataloader(self):
        transforms = Compose([
            ToNumpy(),
            AudioSqueeze(),
            self.audio_transform
        ])
        dataset_train = get_dataset(self.config, part='train', transforms=transforms, keys=['audio', 'sample_rate'])
        dataset_train = torch.utils.data.DataLoader(dataset_train,
                              batch_size=self.batch_size, collate_fn=no_pad_collate, shuffle=True, num_workers=self.num_workers)
        return dataset_train

    def val_dataloader(self):
        transforms = Compose([
            ToNumpy(),
            AudioSqueeze(),
            self.audio_transform
        ])
        dataset_val = get_dataset(self.config, part='val', transforms=transforms, keys=['audio', 'sample_rate'])
        dataset_val = torch.utils.data.DataLoader(dataset_val,
                              batch_size=1, collate_fn=no_pad_collate, num_workers=1)
        return dataset_val
