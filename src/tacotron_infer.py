import argparse
import os
import yaml
from easydict import EasyDict as edict
from tacotron2.model.net import Tacotron2, MusicTacotron
from waveglow import Vocoder as WaveglowVocoder
from wavegrad import OptimizerVocoder, LibrosaVocoder
from easydict import EasyDict as edict
import torchaudio
import torch
from data.transforms import (
    MelSpectrogram, Compose, AddLengths, Pad,
    ToNumpy, AudioSqueeze, ToGpu, AudioEncode
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer model.')
    parser.add_argument('--config', default='tacotron2/configs/ljspeech_tacotron.yml',
                        help='path to config file')
    parser.add_argument('--n_samples', default=100000,
                        help='size of audio')
    parser.add_argument('--weights', help='Weights to be infered')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = edict(yaml.safe_load(stream))
    vocoder = WaveglowVocoder()
    try:
        vocoder.eval()
    except:
        try:
            vocoder.mel.to('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            pass

    model = MusicTacotron(config)
    model.load_state_dict(torch.load(args.weights))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        model.eval()
        mel_outputs, mel_outputs_postnet, alignments = model(
            lengths=torch.tensor(args.n_samples).to('cuda' if torch.cuda.is_available() else 'cpu')
        )
        reconstructed_wav = vocoder.inference(mel_outputs_postnet[0].permute(1, 0)[None])[0]
    torchaudio.save('predicted_audio.wav', reconstructed_wav, sr)
