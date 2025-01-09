from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm


class AudioDataset(Dataset):
    """Dataset for loading and processing audio files."""

    def __init__(
        self,
        base_path: str | Path,
        data_name: str,
        device: torch.device | str = "cuda",
        segment_length: int = 16384,  # ~1 second at 16kHz
        sample_rate: int = 16000,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.path_to_dataset = Path(base_path)
        self.data_name = data_name

        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512,
            win_length=1024,
        )

        self.segments = self._load_dataset()

    def _load_dataset(self):
        """Load and process all audio files."""
        clean_dir = self.path_to_dataset / f"clean_{self.data_name}"
        noisy_dir = self.path_to_dataset / f"noisy_{self.data_name}"

        segments = []
        files = list(clean_dir.glob("*.wav"))
        for clean_path in tqdm(files):
            noisy_path = noisy_dir / clean_path.name
            if noisy_path.exists():
                segments.extend(self._process_audio_file(clean_path, noisy_path))

        return segments

    def _process_audio_file(self, clean_path: Path, noisy_path: Path) -> list:
        """Process a pair of clean and noisy audio files into segments."""
        try:
            clean_waveform, sr = torchaudio.load(str(clean_path))
            noisy_waveform, _ = torchaudio.load(str(noisy_path))

            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                clean_waveform = resampler(clean_waveform)
                noisy_waveform = resampler(noisy_waveform)

            if clean_waveform.size(0) > 1:
                clean_waveform = torch.mean(clean_waveform, dim=0, keepdim=True)
            if noisy_waveform.size(0) > 1:
                noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)

            segments = []
            length = clean_waveform.size(1)

            for start in range(
                0, length - self.segment_length + 1, self.segment_length // 2
            ):
                end = start + self.segment_length

                if end > length:
                    clean_segment = torch.nn.functional.pad(
                        clean_waveform[:, start:],
                        (0, self.segment_length - (length - start)),
                    )
                    noisy_segment = torch.nn.functional.pad(
                        noisy_waveform[:, start:],
                        (0, self.segment_length - (length - start)),
                    )
                else:
                    clean_segment = clean_waveform[:, start:end]
                    noisy_segment = noisy_waveform[:, start:end]

                mel_spec = self.mel_spectrogram_transform(clean_segment)

                fft = torch.fft.fft(clean_segment.squeeze(0))
                fourier = torch.stack([fft.real, fft.imag], dim=0)

                segments.append((clean_segment, mel_spec, fourier, noisy_segment))

            return segments

        except Exception as e:
            print(f"Error processing file {clean_path}: {str(e)}")
            return []

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a segment from the dataset."""
        clean_wave, mel_spec, fourier, noisy_wave = self.segments[idx]

        return (
            clean_wave.to(self.device),
            mel_spec.to(self.device),
            fourier.to(self.device),
            noisy_wave.to(self.device),
        )

    def __len__(self) -> int:
        """Return the number of segments in the dataset."""
        return len(self.segments)


if __name__ == "__main__":
    dataset = AudioDataset("data/", "trainset_200_samples", device="cpu")
    print(f"Dataset size: {len(dataset)}")

    clean_wave, mel_spec, fourier, noisy_wave = dataset[0]
    print("\nTensor shapes:")
    print(f"Clean wave: {clean_wave.shape}")
    print(f"Mel spectrogram: {mel_spec.shape}")
    print(f"Fourier transform: {fourier.shape}")
    print(f"Noisy wave: {noisy_wave.shape}")
