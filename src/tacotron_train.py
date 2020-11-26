import argparse
import yaml
from easydict import EasyDict as edict
from tacotron2 import trainers
from waveglow import Vocoder
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('--config', default='tacotron2/configs/ljspeech_tacotron.yml',
                        help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = edict(yaml.safe_load(stream))
    pl_model = getattr(trainers, config.train.trainer)(config, Vocoder=Vocoder)
    wandb_logger = WandbLogger(name='final_kiss', project=os.basename(args.config).split('.')[0], log_model=True)
    wandb_logger.log_hyperparams(config)
    wandb_logger.watch(pl_model.model, log='all', log_freq=100)
    trainer = pl.Trainer(logger=wandb_logger,
        **config.train.trainer_args)
    trainer.fit(pl_model)
