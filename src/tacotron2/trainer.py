import os
import torch
import wandb
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data.ljspeech import get_dataset
from data.musicnet import MusicNet
from data.electro import ElectroDataset, Electro2Dataset
from data.transforms import (
    MelSpectrogram, Compose, AddLengths, Pad,
    TextPreprocess, ToNumpy, AudioSqueeze, ToGpu)
from torchaudio.transforms import Resample
from data.collate import no_pad_collate
from utils import fix_seeds
from tacotron2.model.net import Tacotron2, MusicTacotron


class Tacotron2Trainer(pl.LightningModule):
    def __init__(
           self,
           config,
           Vocoder=None
        ):
        super(Tacotron2Trainer, self).__init__()
        fix_seeds(seed=config.train.seed)
        self.model = Tacotron2(config)
        self.lr = config.train.lr
        self.batch_size = config.train.batch_size
        self.weight_decay = config.train.get('weight_decay', 0.)
        self.num_workers = config.train.get('num_workers', 4)
        self.step_size = config.train.get('step_size', 15)
        self.gamma =  config.train.get('gamma', 0.2)
        self.text_transform = TextPreprocess(config.alphabet)
        self.mel = MelSpectrogram()
        self.gpu = ToGpu('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocess = Compose([
            AddLengths(),
            Pad()
        ])
        self.mseloss = nn.MSELoss()
        self.gate_bce = nn.BCEWithLogitsLoss()
        self.g = config.train.get(
            'guiding_window_width', 0.2
        )
        if Vocoder is not None:
            self.vocoder = Vocoder().eval()
        else:
            self.vocoder = None
        self.config = config
        self.sample_rate = config.dataset.get('sample_rate', 16000)
        self.epoch_idx = 0

    def forward(self, batch):
        if self.training:
            return self.model(
                text_inputs=batch['text'],
                lengths=batch['text_lengths'],
                mels=batch['mel']
            )
        else:
            return self.model(
                text_inputs=batch['text'])

    def mels_mse(self, mel_outputs, mel_outputs_postnet, batch):
        if self.training:
            y = batch['mel']
            y.requires_grad = False
            batch_size, max_length, n_mel_channels = y.shape
            output_lengths = batch['mel_lengths']
            mask = torch.arange(max_length, device=output_lengths.device,
                            dtype=output_lengths.dtype)[None, :] < output_lengths[:, None]
            mask = mask.bool()
            mask = mask[..., None].repeat_interleave(n_mel_channels, dim=2)
            mask.requires_grad = False
            return self.mseloss(mel_outputs * mask,  y * mask) + self.mseloss(
                        mel_outputs_postnet * mask, y * mask)
        else:
            y = batch['mel'][:, :mel_outputs.shape[1]]
            mel_outputs = mel_outputs[:, :y.shape[1]]
            mel_outputs_postnet = mel_outputs_postnet[:, :y.shape[1]]
            return self.mseloss(mel_outputs, y) + self.mseloss(mel_outputs_postnet, y)

    def guided_attention_loss(self, alignments):
        b, t, n = alignments.shape
        grid_t, grid_n = torch.meshgrid(torch.arange(t, device=alignments.device), torch.arange(n, device=alignments.device))
        W = 1. - torch.exp(-(-grid_n / n + grid_t/t) ** 2 / 2 / self.g**2)
        W.requires_grad = False
        return torch.mean(alignments * W[None].repeat_interleave(b, dim=0)), W

    def gate_loss(self, gate_out, mel_lengths):
        gate_target = torch.zeros_like(gate_out)
        for i in range(gate_out.shape[0]):
            gate_target[i, mel_lengths[i]:] = 1
        gate_target.requires_grad = False
        return self.gate_bce(gate_out, gate_target)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        batch = self.mel(self.gpu(batch))
        batch = self.preprocess(batch)
        batch['mel'] = batch['mel'].permute(0, 2, 1)
        mel_outputs, mel_outputs_postnet, gate_out, alignments = self(batch)
        train_mse = self.mels_mse(mel_outputs, mel_outputs_postnet, batch)
        train_gate = self.gate_loss(gate_out, batch['mel_lengths'])
        loss = train_mse + train_gate
        losses_dict = {
            'train_loss': loss.item(), 'train_mse': train_mse.item(), 'train_gate_loss': train_gate.item()
        }
        if self.config.train.use_guided_attention:
            attn_loss, guide = self.guided_attention_loss(alignments)
            loss += attn_loss
            losses_dict['train_attn_loss'] = attn_loss.item()
        self.logger.experiment.log(losses_dict)
        if batch_nb % self.config.train.train_log_period == 1:
            examples = [
                wandb.Image(mel_outputs_postnet[0].detach().cpu().numpy(), caption='predicted_mel'),
                wandb.Image(batch['mel'][0].detach().cpu().numpy(), caption='target_mel'),
                wandb.Image(alignments[0].detach().cpu().numpy(), caption='alignment')
            ]
            self.logger.experiment.log({'input_texts_train' : wandb.Table(data=[
                    self.text_transform.reverse(batch['text'][0].detach().cpu().numpy())], columns=["Text"])})
            if self.config.train.use_guided_attention:
                examples.append(wandb.Image(guide.cpu().numpy(), caption='attention_guide'))
            self.logger.experiment.log({
                "plots_train": examples
            })
            examples = []
            if self.vocoder is not None:
                reconstructed_wav = self.vocoder.inference(mel_outputs_postnet[0].detach().permute(1, 0)[None])[0]
                examples.append(wandb.Audio(reconstructed_wav.detach().cpu().numpy(), caption='reconstructed_wav', sample_rate=self.sample_rate))
                examples.append(wandb.Audio(batch['audio'][0].detach().cpu().numpy(), caption='target_wav', sample_rate=self.sample_rate))
            self.logger.experiment.log({
                "audios_train": examples
            })
        return loss


    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        batch = self.mel(self.gpu(batch))
        batch = self.preprocess(batch)
        batch['mel'] = batch['mel'].permute(0, 2, 1)
        mel_outputs, mel_outputs_postnet, gate_out, alignments = self(batch)
        mse = self.mels_mse(mel_outputs, mel_outputs_postnet, batch)
        gate = self.gate_loss(gate_out, batch['mel_lengths'])
        loss = mse + gate
        losses_dict = {'val_loss': loss, 'val_mse': mse, 'val_gate_loss': gate}
        if self.config.train.use_guided_attention:
            attn_loss, guide = self.guided_attention_loss(alignments)
            losses_dict['val_attn_loss'] = attn_loss
            loss += attn_loss
        if batch_nb % self.config.train.val_log_period == 1:
            examples = [
                wandb.Image(mel_outputs_postnet[0].cpu().numpy(), caption='predicted_mel'),
                wandb.Image(batch['mel'][0].cpu().numpy(), caption='target_mel'),
                wandb.Image(alignments[0].cpu().numpy(), caption='alignment')
            ]
            self.logger.experiment.log({'input_texts_val' : wandb.Table(data=[
                    self.text_transform.reverse(batch['text'][0].cpu().numpy())], columns=["Text"])})
            self.logger.experiment.log({
                "plots_val": examples
            })
            examples = []
            if self.vocoder is not None:
                reconstructed_wav = self.vocoder.inference(mel_outputs_postnet[0].permute(1, 0)[None])[0]
                examples.append(wandb.Audio(reconstructed_wav.cpu().numpy(), caption='reconstructed_wav', sample_rate=self.sample_rate))
                examples.append(wandb.Audio(batch['audio'][0].cpu().numpy(), caption='target_wav', sample_rate=self.sample_rate))
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
        if self.epoch_idx > 0:
            try:
                os.system('rm ' + os.path.join(self.config.train.get('checkpoint_path', 'checkpoints'), f'model_{self.epoch_idx - 1}.pth'))
            except:
                print('not delete old checkpoint')
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
            self.text_transform,
            ToNumpy(),
            AudioSqueeze()
        ])
        dataset_train = get_dataset(self.config, part='train', transforms=transforms)
        dataset_train = torch.utils.data.DataLoader(dataset_train,
                              batch_size=self.batch_size, collate_fn=no_pad_collate, shuffle=True, num_workers=self.num_workers)
        return dataset_train

    def val_dataloader(self):
        transforms = Compose([
            self.text_transform,
            ToNumpy(),
            AudioSqueeze()
        ])
        dataset_val = get_dataset(self.config, part='val', transforms=transforms)
        dataset_val = torch.utils.data.DataLoader(dataset_val,
                              batch_size=1, collate_fn=no_pad_collate, num_workers=1)
        return dataset_val




class MusicNetTrainer(Tacotron2Trainer):
    def __init__(
           self,
           config,
           Vocoder=None
        ):
        super(Tacotron2Trainer, self).__init__()
        fix_seeds(seed=config.train.seed)
        self.model = MusicTacotron(config)
        self.lr = config.train.lr
        self.batch_size = config.train.batch_size
        self.weight_decay = config.train.get('weight_decay', 0.)
        self.num_workers = config.train.get('num_workers', 4)
        self.step_size = config.train.get('step_size', 15)
        self.gamma =  config.train.get('gamma', 0.2)
        self.mel = MelSpectrogram()
        self.gpu = ToGpu('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocess = Compose([
            AddLengths(),
            Pad()
        ])
        self.mseloss = nn.MSELoss()
        self.gate_bce = nn.BCEWithLogitsLoss()
        self.g = config.train.get(
            'guiding_window_width', 0.2
        )
        if Vocoder is not None:
            self.vocoder = Vocoder()
            try:
                self.vocoder.eval()
            except:
                try:
                    self.vocoder.mel.to('cuda' if torch.cuda.is_available() else 'cpu')
                except Exception as e:
                    pass
        else:
            self.vocoder = None
        self.config = config
        self.sample_rate = config.dataset.get('sample_rate', 16000)
        self.epoch_idx = 0

    def forward(self, batch):
        if self.training:
            return self.model(
                mels=batch['mel'],
                lengths=batch['mel_lengths']
            )
        else:
            return self.model(
                lengths=batch['mel_lengths']
            )

    def training_step(self, batch, batch_nb):
        # REQUIRED
        batch = self.mel(self.gpu(batch))
        batch = self.preprocess(batch)
        batch['mel'] = batch['mel'].permute(0, 2, 1)
        mel_outputs, mel_outputs_postnet, alignments = self(batch)
        train_mse = self.mels_mse(mel_outputs, mel_outputs_postnet, batch)
        loss = train_mse
        losses_dict = {
            'train_loss': loss.item(), 'train_mse': train_mse.item()
        }
        if self.config.train.use_guided_attention:
            attn_loss, guide = self.guided_attention_loss(alignments)
            loss += attn_loss
            losses_dict['train_attn_loss'] = attn_loss.item()
        self.logger.experiment.log(losses_dict)
        if batch_nb % self.config.train.train_log_period == 0:
            examples = [
                wandb.Image(mel_outputs_postnet[0].detach().cpu().numpy(), caption='predicted_mel'),
                wandb.Image(batch['mel'][0].detach().cpu().numpy(), caption='target_mel'),
                wandb.Image(alignments[0].detach().cpu().numpy(), caption='alignment')
            ]
            if self.config.train.use_guided_attention:
                examples.append(wandb.Image(guide.cpu().numpy(), caption='attention_guide'))
            self.logger.experiment.log({
                "plots_train": examples
            })
            examples = []
            if self.vocoder is not None:
                reconstructed_wav = self.vocoder.inference(mel_outputs_postnet[0].detach().permute(1, 0)[None])[0]
                target_wav_vocoded = self.vocoder.inference(batch['mel'][0].permute(1, 0)[None])[0]
                examples.append(wandb.Audio(reconstructed_wav.cpu().numpy(), caption='reconstructed_wav', sample_rate=self.sample_rate))
                examples.append(wandb.Audio(target_wav_vocoded.cpu().numpy(), caption='target_wav_vocoded', sample_rate=self.sample_rate))
                examples.append(wandb.Audio(batch['audio'][0].detach().cpu().numpy(), caption='target_wav', sample_rate=self.sample_rate))
            self.logger.experiment.log({
                "audios_train": examples
            })
        return loss


    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        batch = self.mel(self.gpu(batch))
        batch = self.preprocess(batch)
        batch['mel'] = batch['mel'].permute(0, 2, 1)
        mel_outputs, mel_outputs_postnet, alignments = self(batch)
        mse = self.mels_mse(mel_outputs, mel_outputs_postnet, batch)
        loss = mse
        losses_dict = {'val_loss': loss, 'val_mse': mse}
        if self.config.train.use_guided_attention:
            attn_loss, guide = self.guided_attention_loss(alignments)
            losses_dict['val_attn_loss'] = attn_loss
            loss += attn_loss
        if batch_nb % self.config.train.val_log_period == 0:
            examples = [
                wandb.Image(mel_outputs_postnet[0].cpu().numpy(), caption='predicted_mel'),
                wandb.Image(batch['mel'][0].cpu().numpy(), caption='target_mel'),
                wandb.Image(alignments[0].cpu().numpy(), caption='alignment')
            ]
            self.logger.experiment.log({
                "plots_val": examples
            })
            examples = []
            if self.vocoder is not None:
                reconstructed_wav = self.vocoder.inference(mel_outputs_postnet[0].permute(1, 0)[None])[0]
                target_wav_vocoded = self.vocoder.inference(batch['mel'][0].permute(1, 0)[None])[0]
                examples.append(wandb.Audio(reconstructed_wav.cpu().numpy(), caption='reconstructed_wav', sample_rate=self.sample_rate))
                examples.append(wandb.Audio(target_wav_vocoded.cpu().numpy(), caption='target_wav_vocoded', sample_rate=self.sample_rate))
                examples.append(wandb.Audio(batch['audio'][0].cpu().numpy(), caption='target_wav', sample_rate=self.sample_rate))
            self.logger.experiment.log({
                "audios_val": examples
            })
        return losses_dict

    def prepare_data(self):
        MusicNet(root=self.config.dataset.root, download=True)

    def train_dataloader(self):
        transforms = Compose([
            ToNumpy()
            # Resample(44100, 22050),
            # AudioSqueeze()
        ])
        dataset_train = MusicNet(root=self.config.dataset.root, train=True, pitch_shift=self.config.dataset.get('pitch_shift', 0.), jitter=self.config.dataset.get('jitter', 0.), transforms=transforms)
        dataset_train = torch.utils.data.DataLoader(dataset_train,
                              batch_size=self.batch_size, collate_fn=no_pad_collate, shuffle=True, num_workers=self.num_workers)
        return dataset_train

    def val_dataloader(self):
        transforms = Compose([
            ToNumpy()
            # Resample(44100, 22050),
            # AudioSqueeze()
        ])
        dataset_val = MusicNet(root=self.config.dataset.root, train=False, transforms=transforms, epoch_size=self.config.train.get('val_size', 1000))
        dataset_val = torch.utils.data.DataLoader(dataset_val,
                              batch_size=1, collate_fn=no_pad_collate, num_workers=1)
        return dataset_val

class ElectroTrainer(MusicNetTrainer):
    def prepare_data(self):
        pass

    def train_dataloader(self):
        transforms = Compose([
            ToNumpy()
            # Resample(44100, 16000),
            # AudioSqueeze()
        ])
        dataset_train = Electro2Dataset(root=self.config.dataset.root, subfolder='train', transforms=transforms)
        dataset_train = torch.utils.data.DataLoader(dataset_train,
                              batch_size=self.batch_size, collate_fn=no_pad_collate, shuffle=True, num_workers=self.num_workers)
        return dataset_train

    def val_dataloader(self):
        transforms = Compose([
            ToNumpy()
            # Resample(44100, 16000),
            # AudioSqueeze()
        ])
        dataset_val = Electro2Dataset(root=self.config.dataset.root, subfolder='val', transforms=transforms)
        dataset_val = torch.utils.data.DataLoader(dataset_val,
                              batch_size=1, collate_fn=no_pad_collate, num_workers=1)
        return dataset_val
