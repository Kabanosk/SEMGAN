from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Class for loading and processing the audio dataset."""

    def __init__(
        self, base_path: str | Path, data_name: str, device: torch.device | str = "cuda"
    ):
        self.device: str = device
        self.sample_rate: int = 16000
        self.max_length: int = self.sample_rate * 10  # Max length for padding (e.g., 10 seconds for 16kHz)
        self.path_to_dataset: Path = Path(base_path)
        self.data_name: str | Path = data_name

        # Transforms
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
        )

        self._load_dataset()

    def _load_dataset(self):
        """Load audio files and compute features."""
        clean_trainset_dir = self.path_to_dataset / f"clean_{self.data_name}"
        noisy_trainset_dir = self.path_to_dataset / f"noisy_{self.data_name}"

        self.waveform_dataset: list = []
        self.mel_spectrogram_dataset: list = []
        self.fourier_transform_dataset: list = []
        self.noisy_waveform_dataset: list = []

        # Loop through clean and noisy audio files and load them
        for clean_wav_file in clean_trainset_dir.glob("*.wav"):
            noisy_wav_file = noisy_trainset_dir / clean_wav_file.name  # Get corresponding noisy file

            if noisy_wav_file.exists():  # Ensure the noisy file exists
                clean_waveform, sr = torchaudio.load(clean_wav_file)
                if sr != self.sample_rate:
                    resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                    clean_waveform = resampler(clean_waveform)

                noisy_waveform, sr = torchaudio.load(noisy_wav_file)
                if sr != self.sample_rate:
                    resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                    noisy_waveform = resampler(noisy_waveform)

                # Padding if necessary to match max length
                clean_waveform = self._pad_audio(clean_waveform)
                noisy_waveform = self._pad_audio(noisy_waveform)

                self.waveform_dataset.append(clean_waveform)
                self.noisy_waveform_dataset.append(noisy_waveform)

                # Compute Mel Spectrogram
                mel_spectrogram = self.mel_spectrogram_transform(clean_waveform)
                self.mel_spectrogram_dataset.append(mel_spectrogram)

                # Compute Fourier Transform (Magnitude Spectrum)
                fourier_transform = torch.fft.fft(clean_waveform)
                magnitude = torch.abs(fourier_transform)  # Take magnitude
                self.fourier_transform_dataset.append(magnitude)

    def _pad_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Pad audio tensor to max length."""
        if audio_tensor.size(1) < self.max_length:
            padding_length = self.max_length - audio_tensor.size(1)
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding_length))
        return audio_tensor

    def __getitem__(self, idx):
        return (
            self.waveform_dataset[idx].to(self.device),
            self.mel_spectrogram_dataset[idx].to(self.device),
            self.fourier_transform_dataset[idx].to(self.device),
            self.noisy_waveform_dataset[idx].to(self.device),
        )

    def __len__(self):
        return len(self.waveform_dataset)


if __name__ == "__main__":
    dataset = AudioDataset("data/", "testset_wav", device="cpu")
    print(f"Number of samples: {len(dataset)}")
    waveform, mel_spectrogram, fourier_transform, _ = dataset[0]
    print(f"Waveform shape: {waveform.shape}")
    print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
    print(f"Fourier transform shape: {fourier_transform.shape}")
