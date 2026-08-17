#!/usr/bin/env python3
"""Generate sample WAV files for benchmarking the voice pipeline.

Uses espeak-ng to synthesize speech samples for each of the 8 intents.
Run this inside the Docker container where espeak-ng is available.

Usage:
    docker compose exec pipeline_service python generate_samples.py
    python generate_samples.py  # if espeak-ng is installed locally
"""

import os
import subprocess
import shutil
import sys

SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "samples")

SAMPLE_COMMANDS = [
    ("TurnOn", "turn on the lights"),
    ("TurnOn", "turn on the fan"),
    ("TurnOn", "switch on the air conditioner"),
    ("TurnOff", "turn off the lights"),
    ("TurnOff", "turn off the fan"),
    ("TurnOff", "switch off the television"),
    ("SetBrightness", "set brightness to 50 percent"),
    ("SetBrightness", "dim the lights to 30"),
    ("SetBrightness", "make it brighter"),
    ("GetTemperature", "what is the temperature"),
    ("GetTemperature", "how hot is it outside"),
    ("GetTemperature", "tell me the room temperature"),
    ("PlayMusic", "play music"),
    ("PlayMusic", "play some tunes"),
    ("PlayMusic", "start playing my playlist"),
    ("StopMusic", "stop the music"),
    ("StopMusic", "pause the music"),
    ("StopMusic", "stop playing"),
    ("SetTimer", "set a timer for 5 minutes"),
    ("SetTimer", "set a timer for 10 minutes"),
    ("SetTimer", "start a timer"),
    ("GetWeather", "what is the weather"),
    ("GetWeather", "how is the weather outside"),
    ("GetWeather", "check the forecast"),
]


def main():
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        print("ERROR: espeak-ng or espeak not found.")
        print("Install with: apt-get install -y espeak-ng")
        sys.exit(1)

    print(f"Using TTS: {espeak}")
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    for i, (intent, phrase) in enumerate(SAMPLE_COMMANDS, 1):
        filename = os.path.join(SAMPLES_DIR, f"sample_{i:02d}.wav")
        cmd = [
            espeak,
            "-v", "en-us",
            "-s", "150",
            "-w", filename,
            phrase,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"  [{i:02d}] {intent}: '{phrase}' -> {filename}")

    print(f"\nGenerated {len(SAMPLE_COMMANDS)} sample WAV files in {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
