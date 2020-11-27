# TTS
Text to speech on pytorch.
Implementation of [tacotron-2](https://arxiv.org/pdf/1712.05884.pdf) and [wavenet](https://arxiv.org/pdf/1609.03499.pdf).
Based on: [dl-start-pack](https://github.com/markovka17/dl-start-pack).

Tacotron-2 core features:
 - Pytorch-lightning training from various configs: src/tacotron2/configs
 - Wandb Logging
 - [Guided attention](https://arxiv.org/pdf/1710.08969.pdf)
 - [Monotonic attention](https://arxiv.org/pdf/1704.00784.pdf)
 - Teacher forcing value choose (by default 1. - fully teacher forced)
 - LJspeech dataset training


Several results && weights:
 - [Default tacotron-2](https://wandb.ai/oleges/ljspeech_tacotron/runs/21uk4e0i)
 - [Tacotron2 Guided Attention](https://wandb.ai/oleges/ljspeech_tacotron_guided/runs/2ybsmorn)
 - [Tacotron2 Monotonic Attention](https://wandb.ai/oleges/ljspeech_tacotron_monotonic/runs/25lj2aqa)
