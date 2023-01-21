import numpy as np
import os
import torch
import torchaudio
from literai.util import get_output_dir
from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_voices
from tqdm import tqdm

SAMPLE_RATE = 24000


def record_podcast(title: str, alice_voice: str, bob_voice: str, save_recorded_lines=False):
    scripts = get_output_dir(title, "scripts")
    parts = [os.path.join(scripts, f) for f in os.listdir(
        scripts) if f.startswith("part") and f.endswith(".txt")]

    tts = TextToSpeech(models_dir=MODELS_DIR)

    alice_voice_samples, alice_conditioning_latents = load_voices([
                                                                  alice_voice])
    bob_voice_samples, bob_conditioning_latents = load_voices([bob_voice])

    # 250 ms of silence
    silence = torch.zeros(
        1, int(SAMPLE_RATE / 4), dtype=torch.float32, device=torch.device('cpu'))

    if save_recorded_lines:
        recorded_lines = get_output_dir(title, "recorded-lines")
        torchaudio.save(os.path.join(
            recorded_lines, f"silence.wav"), silence, SAMPLE_RATE)
    else:
        recorded_lines = None

    recorded = get_output_dir(title, "recorded")

    for part in tqdm(parts, desc="Part", leave=False):
        lines = open(part, "r", encoding="utf8").readlines()

        part_base = os.path.basename(part)
        part_base = part_base[0:part_base.rfind('.')]

        podcast = []
        alice_count = 0
        bob_count = 0

        for line in tqdm(lines, desc="Line", leave=False):
            colon = line.find(':')
            speaker = line[0:colon]
            text = line[colon+1:].strip()
            if len(text) == 0:
                podcast.extend([silence, silence, silence])
                continue

            if speaker == 'Alice':
                voice_samples = alice_voice_samples
                conditioning_latents = alice_conditioning_latents
                speaker_count = alice_count
                alice_count += 1
            elif speaker == 'Bob':
                voice_samples = bob_voice_samples
                conditioning_latents = bob_conditioning_latents
                speaker_count = bob_count
                bob_count += 1
            else:
                raise RuntimeError(f"Unknown speaker {speaker}")

            split_text = []
            while len(text) > 0:
                if len(text) <= 256:
                    split_text.append(text)
                    text = ""
                else:
                    i = 256
                    while i < len(text):
                        if text[i] == ' ':
                            break
                        else:
                            i += 1
                    split_text.append(text[0:i])
                    text = text[i:]

            if len(podcast) > 0:
                podcast.append(silence)

            full_line = []
            for to_say in split_text:
                gen = tts.tts_with_preset(to_say, voice_samples=voice_samples,
                                          conditioning_latents=conditioning_latents, preset='standard', k=1, verbose=False).squeeze(0).cpu()
                full_line.append(gen)
                podcast.append(gen)

            if recorded_lines is not None:
                if len(full_line) == 1:
                    recorded_line = full_line[0]
                else:
                    recorded_line = torch.cat(full_line, dim=-1)
                torchaudio.save(os.path.join(
                    recorded_lines, f"{part_base}_{speaker.lower()}{speaker_count}.wav"), recorded_line, SAMPLE_RATE)

        full_audio = torch.cat(podcast, dim=-1)
        torchaudio.save(os.path.join(
            recorded, f"{part_base}.wav"), full_audio, SAMPLE_RATE)
