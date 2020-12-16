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

    model = MusicTacotron(**config.model)
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
        mel_outputs, mel_outputs_postnet, alignments = model(
            lengths=n_samples
        )
        reconstructed_wav = vocoder.inference(mel_outputs_postnet[0].permute(1, 0)[None])[0]
    torchaudio.save('predicted_audio.wav', reconstructed_wav, sr)
