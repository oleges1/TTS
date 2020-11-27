#!/bin/bash

image_name=tts_pytorch

docker build -t "{$image_name}":latest ..
