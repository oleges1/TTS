# never tested
FROM kaggle/python

# WORKDIR /home/user

# Requirements
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

CMD cd src && python3 tacotron_train.py --config=tacotron2/configs/ljspeech_tacotron_monotonic.yaml
