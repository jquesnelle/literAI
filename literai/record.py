import functools
import json
import os
import torch
import torchaudio
from literai.util import get_output_dir, logger_error
from pydub import AudioSegment
from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_voices
from tqdm import tqdm
from typing import Dict, List, Optional

SAMPLE_RATE = 24000
MAX_SPEECH_LEN = 192
TAGLINE = "Happy singularity!"


def duration(speech: torch.tensor) -> float:
    return speech.shape[1] / SAMPLE_RATE


@logger_error
def record_podcast(title: str, voices: List[str], save_recorded_lines=False, single_part: Optional[str] = None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    base_dir = get_output_dir(title)
    parts = [os.path.join(base_dir, f) for f in os.listdir(
        base_dir) if f.startswith("part" if single_part is None else f"{single_part}.") and f.endswith(".json")]

    tts = TextToSpeech(models_dir=MODELS_DIR)

    voice_data = [load_voices([voice]) for voice in voices]

    # 250 ms of silence
    silence = torch.zeros(
        1, int(SAMPLE_RATE / 4), dtype=torch.float32, device=torch.device('cpu'))
    silence_duration = duration(silence)

    taglines = [tts.tts_with_preset(TAGLINE, voice_samples=voice_data[speaker][0],
                                    conditioning_latents=voice_data[speaker][1], preset='standard', k=1, verbose=False).squeeze(0).cpu() for speaker in range(0, len(voices))]

    max_ending_dimension = max([tagline.shape[1] for tagline in taglines])

    # sum all taglines together, padded with zeroes to the length of the longest
    endings = [torch.zeros(1, max_ending_dimension,
                           dtype=torch.float32, device=torch.device('cpu'))]
    endings.extend([torch.nn.functional.pad(tagline, pad=(
        0, max_ending_dimension - tagline.shape[1])) for tagline in taglines])
    ending = functools.reduce(lambda a, b: a+b, endings)

    if save_recorded_lines:
        recorded_lines = get_output_dir(title, "recorded-lines")
        torchaudio.save(os.path.join(
            recorded_lines, f"silence.wav"), silence, SAMPLE_RATE)
        torchaudio.save(os.path.join(
            recorded_lines, f"ending.wav"), ending, SAMPLE_RATE)
    else:
        recorded_lines = None

    recorded = get_output_dir(title, "recorded")

    for part in tqdm(parts, desc="Part"):
        obj = json.load(open(part, "r", encoding="utf8"))

        part_base = os.path.basename(part)
        part_base = part_base[0:part_base.rfind('.')]

        podcast = []
        podcast_duration = 0.0
        speakers_counts: Dict[int, int] = {}

        for line in tqdm(obj['lines'], desc="Line", leave=False):
            text = line['text'].strip()

            speaker = line['speaker']
            voice_samples, conditioning_latents = voice_data[speaker]
            if speaker in speakers_counts:
                speakers_counts[speaker] += 1
            else:
                speakers_counts[speaker] = 1

            split_text = []
            # I'm sure there's a more elegant way to do this
            while len(text) > 0:
                if len(text) <= MAX_SPEECH_LEN:
                    split_text.append(text)
                    text = ""
                else:
                    i = MAX_SPEECH_LEN
                    while i < len(text):
                        if text[i] == ' ':
                            break
                        else:
                            i += 1
                    split_text.append(text[0:i])
                    text = text[i:]

            if len(podcast) > 0:
                podcast.append(silence)
                podcast_duration += silence_duration

            full_line = []
            if len(split_text) > 0 and len(split_text[0]) > 0:
                for to_say in split_text:
                    gen = tts.tts_with_preset(to_say, voice_samples=voice_samples,
                                              conditioning_latents=conditioning_latents, preset='standard', k=1, verbose=False).squeeze(0).cpu()
                    full_line.append(gen)
                    podcast.append(gen)
            else:
                full_line = [silence, silence]
                gen = torch.cat(full_line, dim=-1)

            line['start'] = podcast_duration
            podcast_duration += duration(gen)
            line['end'] = podcast_duration

            if recorded_lines is not None:
                if len(full_line) == 1:
                    recorded_line = full_line[0]
                else:
                    recorded_line = torch.cat(full_line, dim=-1)

                filename = f"{part_base}-{speaker}-{speakers_counts[speaker] - 1}"
                wav_path = os.path.join(recorded_lines, filename) + ".wav"
                torchaudio.save(wav_path, recorded_line, SAMPLE_RATE)

                AudioSegment.from_file(wav_path, format="wav").export(os.path.join(
                    recorded_lines, filename) + ".mp3", format="mp3", bitrate="192k")
                os.remove(wav_path)

                line['audio'] = f"recorded-lines/{filename}.mp3"

        podcast.append(ending)
        podcast_duration += duration(ending)
        obj['duration'] = podcast_duration

        full_audio = torch.cat(podcast, dim=-1)

        wav_path = os.path.join(recorded, part_base) + ".wav"
        torchaudio.save(wav_path, full_audio, SAMPLE_RATE)

        AudioSegment.from_file(wav_path, format="wav").export(os.path.join(
            recorded, part_base) + ".mp3", format="mp3", bitrate="192k")
        os.remove(wav_path)

        obj['audio'] = f"recorded/{part_base}.mp3"

        json.dump(obj, open(part, "w", encoding="utf8"), indent=2)
