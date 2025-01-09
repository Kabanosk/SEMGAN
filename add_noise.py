import os

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def add_gaussian_noise(audio, snr_db):
    sig_power = np.mean(audio**2)
    noise_power = sig_power / (10 ** (snr_db / 10))

    # Generate noise
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    noisy_audio = audio + noise

    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val

    return noisy_audio


def process_directory(input_dir, output_dir, snr_db=20):
    os.makedirs(output_dir, exist_ok=True)
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    for wav_file in tqdm(wav_files):
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, wav_file)

        sample_rate, audio = wavfile.read(input_path)
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max

        # Add noise
        noisy_audio = add_gaussian_noise(audio, snr_db)
        noisy_audio = (noisy_audio * np.iinfo(np.int16).max).astype(np.int16)

        wavfile.write(output_path, sample_rate, noisy_audio)


if __name__ == "__main__":
    input_dir = "data/clean_trainset_1spk_wav"
    output_dir = "data/noisy_trainset_1spk_wav"
    snr_db = 20

    process_directory(input_dir, output_dir, snr_db)
