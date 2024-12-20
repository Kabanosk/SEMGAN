"""File for loading and processing the audio dataset."""

from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Class for loading and processing the audio dataset."""

    def __init__(self, base_path: str | Path, data_name: str, device: torch.device | str = "cuda"):
        self.waveform_dataset: list = []
        self.mel_spectrogram_dataset: list = []
        self.fourier_transform_dataset: list = []
        self.device: str = device

        self.path_to_dataset: Path = Path(base_path)
        self.data_name: str | Path = data_name
        self._load_dataset()

    def _load_dataset(self):
        """Load audio files and compute features."""
        clean_trainset_dir = self.path_to_dataset / f"clean_{self.data_name}"
        noisy_trainset_dir = self.path_to_dataset / f"noisy_{self.data_name}"

        # Loop through clean and noisy audio files and load them
        for clean_wav_file in clean_trainset_dir.glob("*.wav"):
            noisy_wav_file = (
                noisy_trainset_dir / clean_wav_file.name
            )  # Get corresponding noisy file

            if noisy_wav_file.exists():  # Ensure the noisy file exists
                clean_waveform, sr = torchaudio.load(clean_wav_file)
                noisy_waveform, _ = torchaudio.load(noisy_wav_file)

                self.waveform_dataset.append(clean_waveform)

                mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=sr
                )
                mel_spectrogram = mel_spectrogram_transform(clean_waveform)
                self.mel_spectrogram_dataset.append(mel_spectrogram)

                # Compute Fourier transform
                fourier_transform = torch.fft.fft(clean_waveform)
                self.fourier_transform_dataset.append(fourier_transform)

    def __getitem__(self, idx):
        return (
            self.waveform_dataset[idx].to(self.device),
            self.mel_spectrogram_dataset[idx].to(self.device),
            self.fourier_transform_dataset[idx].to(self.device),
        )

    def __len__(self):
        return len(self.waveform_dataset)


if __name__ == "__main__":
    dataset = AudioDataset("data/", "testset_wav", device="cpu")
    print(f"Number of samples: {len(dataset)}")
    waveform, mel_spectrogram, fourier_transform = dataset[0]
    print(f"Waveform shape: {waveform.shape}")
    print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
    print(f"Fourier transform shape: {fourier_transform.shape}")
