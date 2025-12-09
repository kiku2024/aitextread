from fastapi import FastAPI
from fastapi.responses import FileResponse
import torch
import torchaudio

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

import os
import uuid

app = FastAPI()

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

default_wav, sr = torchaudio.load("assets/exampleaudio.mp3")
default_speaker = model.make_speaker_embedding(default_wav, sr)

@app.get("/tts")
def tts_endpoint(text: str = "テスト"):
    """
    URL例:
    http://UBUNTU_IP:8000/tts?text=テスト
    """

    # Conditioning
    cond_dict = make_cond_dict(
        text=text,
        speaker=default_speaker,
        language="ja"
    )

    conditioning = model.prepare_conditioning(cond_dict)

    codes = model.generate(conditioning)
    wavs = model.autoencoder.decode(codes).cpu()

    out_name = f"tts_{uuid.uuid4()}.wav"
    out_path = f"/tmp/{out_name}"

    torchaudio.save(out_path, wavs[0], model.autoencoder.sampling_rate)

    return FileResponse(out_path, media_type="audio/wav", filename="aispeak.wav")