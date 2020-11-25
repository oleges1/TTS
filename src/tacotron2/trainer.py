import torch
import wandb
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from tacotron2.data.ljspeech import get_dataset
from tacotron2.data.transforms import (
    MelSpectrogram, Compose, AddLengths, Pad, 
    TextPreprocess, ToNumpy, AudioSqueeze, ToGpu)
from tacotron2.data.collate import no_pad_collate
from tacotron2.model.net import Tacotron2
from tacotron2.utils import fix_seeds


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
        self.text_transform = TextPreprocess(config.alphabet)
        self.mel = MelSpectrogram()
        self.gpu = ToGpu('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_preprocess = Compose([
            AddLengths(),
            Pad()
        ])
        self.val_preprocces = AddLengths()
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

    def forward(self, batch):
        batch = self.mel(self.gpu(batch))
        if self.training:
            batch = self.train_preprocess(batch)
        else:
            batch = self.val_preprocces(batch)
        if self.training:
            return self.model(
                text_inputs=batch['text'],
                lengths=batch['text_lengths'],
                mels=batch['mel'],
                output_lengths=batch['mel_lengths']
            )
        else:
            return self.model(
                text_inputs=batch['text'])

    def guided_attention_loss(self, alignments):
        b, n, t = alignments.shape
        grid_n, grid_t = torch.meshgrid(torch.arange(n), torch.arange(t))
        W = 1 - torch.exp(-(grid_n / n - grid_t/t) ** 2 / 2 / g**2)
        return torch.mean(alignments * W[None]), W

    def monotonic_attention_loss(self, alignments):
        # TODO
        return 0

    def gate_loss(self, gate_out, mel_lengths):
        gate_target = torch.zeros(gate_out.shape, dtype='float32', device=mel_lengths.device)
        for i in range(gate_out.shape[0]):
            gate_target[i, mel_lengths[i]:] = 1
        return self.gate_bce(gate_out, gate_target)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self(batch)
        y = batch['mel']
        train_mse = self.mseloss(mel_outputs, y) + self.mseloss(mel_outputs_postnet, y)
        train_gate = self.gate_loss(gate_out, batch['mel_lengths'])
        if self.config.train.use_guided_attention:
            attn_loss, guide = self.guided_attention_loss(alignments)
        elif self.config.train.use_monotonic_attention:
            attn_loss = self.monotonic_attention_loss(alignments)
        loss = train_mse + train_gate + attn_loss
        self.logger.experiment.log({
            'train_loss': loss, 'train_mse': train_mse, 'train_gate_loss': train_gate, 'train_attn_loss': attn_loss
        })
        if batch_nb % self.config.train.train_log_period == 1:
            examples = [
                wandb.Image(mel_outputs_postnet[0].cpu().numpy(), caption='predicted_mel'),
                wandb.Image(y[0].cpu().numpy(), caption='target_mel'),
                wandb.Image(alignments[0].cpu().numpy(), caption='alignment'),
                wandb.Image(gate_out[0].cpu().numpy(), caption='gate')
            ]
            if self.config.train.use_guided_attention:
                examples.append(wandb.Image(guide.cpu().numpy(), caption='attention_guide'))
            if self.vocoder is not None:
                reconstructed_wav = self.vocoder.inference(mel_outputs_postnet[0])
                examples.append(wandb.Audio(reconstructed_wav.cpu().numpy(), caption='reconstructed_wav'))
                examples.append(wandb.Audio(batch['audio'][0].cpu().numpy(), caption='target_wav'))
            self.logger.experiment.log({
                "examples_train": examples
            })
        return {'train_loss': loss}


    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self(batch)
        y = batch['mel']
        mse = self.mseloss(mel_outputs, y) + self.mseloss(mel_outputs_postnet, y)
        gate = self.gate_loss(gate_out, batch['mel_lengths'])
        if self.config.train.use_guided_attention:
            attn_loss, guide = self.guided_attention_loss(alignments)
        elif self.config.train.use_monotonic_attention:
            attn_loss = self.monotonic_attention_loss(alignments)
        loss = mse + gate + attn_loss
        if batch_nb % self.config.train.val_log_period == 1:
            exmaples = [
                wandb.Image(mel_outputs_postnet[0].cpu().numpy(), caption='predicted_mel'),
                wandb.Image(y[0].cpu().numpy(), caption='target_mel'),
                wandb.Image(alignments[0].cpu().numpy(), caption='alignment'),
                wandb.Image(gate_out[0].cpu().numpy(), caption='gate')
            ]
            if self.vocoder is not None:
                reconstructed_wav = self.vocoder.inference(mel_outputs_postnet[0])
                examples.append(wandb.Audio(reconstructed_wav.cpu().numpy(), caption='reconstructed_wav'))
                examples.append(wandb.Audio(batch['audio'][0].cpu().numpy(), caption='target_wav'))
            self.logger.experiment.log({
                "examples_val": exmaples
            })
        return {'val_loss': loss, 'val_mse': mse, 'val_gate_loss': gate, 'val_attn_loss': attn_loss}

    def validation_epoch_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_mse = torch.stack([x['val_mse'] for x in outputs]).mean()
        avg_gate = torch.stack([x['val_gate_loss'] for x in outputs]).mean()
        avg_attn = torch.stack([x['val_attn_loss'] for x in outputs]).mean()

        self.logger.experiment.log({
            'avg_val_loss': avg_loss, 'avg_val_mse': avg_mse,
            'avg_gate_loss': avg_gate, 'avg_val_attn': avg_attn
        })

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
                              batch_size=self.batch_size, collate_fn=no_pad_collate, shuffle=True, num_workers=4)
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
