import argparse
import os
import yaml
from easydict import EasyDict as edict
import torchaudio
import torch
from data.transforms import (
    MelSpectrogram, Compose, AddLengths, Pad,
    ToNumpy, AudioSqueeze, ToGpu, AudioEncode
)
from wavenet.model.net import WaveNet
from utils import fix_seeds, mu_law_decode_torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training model.')
    parser.add_argument('--config', default='wavenet/configs/ljspeech_wavenet.yml',
                        help='path to config file')
    parser.add_argument('--weights', help='path to model weights file')
    parser.add_argument('--audio', help='path to audio file')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = edict(yaml.safe_load(stream))
    model = WaveNet(**config.model)
    model.load_state_dict(torch.load(args.weights))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    audio_transform = AudioEncode(quantization_channels=config.model.get('n_classes', 15))
    mel = MelSpectrogram().to('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = ToGpu('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess = Pad()
    wav, sr = torchaudio.load(args.audio)
    batch = audio_transform({'audio': wav, 'sample_rate': sr})
    batch = mel(gpu({k : [v] for k, v in batch.items()}))
    with torch.no_grad():
        model.eval()
        predictions = model.generate(
            x=torch.empty((1, config.model.get('n_classes', 15), 0), device=batch['audio_quantized'][0].device).float(),
            h=batch['mel'][0],
            samples=batch['audio_quantized'][0].shape[-1]
        )
    predicted_audio = mu_law_decode_torch(predictions[0]).cpu().numpy()
    torchaudio.save('predicted_audio.wav', predicted_audio, sr)
