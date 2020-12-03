import argparse
import os
import yaml
from easydict import EasyDict as edict
from wavenet import trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('--config', default='wavenet/configs/ljspeech_wavenet.yml',
                        help='path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = edict(yaml.safe_load(stream))
    pl_model = getattr(trainer, config.train.trainer)(config)
    wandb_logger = WandbLogger(name='final_kiss',project=os.path.basename(config_path).split('.')[0], log_model=True)
    wandb_logger.log_hyperparams(config)
    wandb_logger.watch(pl_model.model, log='all', log_freq=100)
    trainer = pl.Trainer(logger=wandb_logger,
        **config.train.trainer_args)
    trainer.fit(pl_model)
