"""Script for running inference with trained SEGAN/SEMGAN models."""

import argparse
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from src.utils.model import get_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on audio files")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["segan", "semgan"],
        required=True,
        help="Model architecture (segan or semgan)",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input audio file or directory",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Target sample rate"
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=16384,
        help="Audio segment length for processing",
    )
    parser.add_argument(
        "--clean_file", type=str, help="Path to clean reference file for benchmarking"
    )
    return parser.parse_args()


def process_audio(
    generator: torch.nn.Module,
    audio_path: Path,
    output_path: Path,
    sample_rate: int,
    segment_length: int,
    device: torch.device,
) -> Path:
    """Process a single audio file through the generator with overlapping windows."""
    # Load and preprocess audio
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    if waveform.size(0) > 1:  # Convert stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Process audio in overlapping segments
    generator.eval()

    # Initialize output tensor
    output_waveform = torch.zeros(waveform.size())
    overlap_counter = torch.zeros(waveform.size())

    with torch.no_grad():
        for start in range(
            0, waveform.size(1) - segment_length + 1, segment_length // 2
        ):
            end = start + segment_length
            segment = waveform[:, start:end]

            # Process segment
            segment = segment.to(device)
            if segment.dim() == 2:
                segment = segment.unsqueeze(0)
            z = torch.randn(segment.size(0), 1024, 8).to(device)
            enhanced = generator(segment, z)
            enhanced = enhanced.cpu()

            # Add to output using overlapping windows
            output_waveform[:, start:end] += enhanced.squeeze(0)
            overlap_counter[:, start:end] += 1

        # Handle the last segment if needed
        if end < waveform.size(1):
            start = waveform.size(1) - segment_length
            segment = waveform[:, start:]

            segment = segment.to(device)
            if segment.dim() == 2:
                segment = segment.unsqueeze(0)
            z = torch.randn(segment.size(0), 1024, 8).to(device)
            enhanced = generator(segment, z)
            enhanced = enhanced.cpu()

            output_waveform[:, start:] += enhanced.squeeze(0)
            overlap_counter[:, start:] += 1

    # Average overlapping segments
    output_waveform = output_waveform / overlap_counter.clamp(min=1)

    # Save enhanced audio
    output_file = output_path / f"enhanced_{audio_path.name}"
    torchaudio.save(str(output_file), output_waveform, sample_rate)

    return output_file


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and model
    generator, _ = get_model(
        args.model,
        checkpoint=args.checkpoint,
    )

    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process files
    if input_path.is_file():
        audio_files = [input_path]
    else:
        audio_files = list(input_path.glob("*.wav"))

    pbar = tqdm(audio_files)
    for audio_file in pbar:
        pbar.set_description(f"Processing {audio_file}")
        enhanced_file = process_audio(
            generator,
            audio_file,
            output_path,
            args.sample_rate,
            args.segment_length,
            device,
        )


if __name__ == "__main__":
    main()
