"""File for different audio benchmark functions."""

import numpy as np
from pesq import pesq
from pystoi import stoi
from scipy.io import wavfile


def calculate_pesq(
    clean_signal: np.ndarray, enhanced_signal: np.ndarray, fs: int
) -> float:
    """Calculate PESQ score.

    Args:
        clean_signal (np.array): Clean reference speech signal.
        enhanced_signal (np.array): Enhanced speech signal.
        fs (int): Sampling rate.

    Returns:
        pesq_score (float): PESQ score.
    """
    pesq_score: float = pesq(fs, clean_signal, enhanced_signal)
    return pesq_score


def calculate_stoi(clean_signal: np.array, enhanced_signal: np.array, fs: int) -> float:
    """Calculate STOI score.

    Args:
        clean_signal (np.array): Clean reference speech signal.
        enhanced_signal (np.array): Enhanced speech signal.
        fs (int): Sampling rate.

    Returns:
        stoi_score (float): STOI score.
    """
    stoi_score: float = stoi(clean_signal, enhanced_signal, fs, extended=False)
    return stoi_score


def benchmark(clean_wav_path: str, enhanced_wav_path: str) -> (float, float):
    """Benchmark using PESQ and STOI.

    Args:
        clean_wav_path (str): Path to the clean reference .wav file.
        enhanced_wav_path (str): Path to the enhanced .wav file.

    Returns:
        pesq_score (float): PESQ score.
        stoi_score (float): STOI score.

    Raises:
        ValueError: If sampling rates do not match.
    """
    fs_clean, clean_signal = wavfile.read(clean_wav_path)
    fs_enhanced, enhanced_signal = wavfile.read(enhanced_wav_path)

    if fs_clean != fs_enhanced:
        raise ValueError("Sampling rates of clean and enhanced signals do not match")

    clean_signal = clean_signal.astype(np.float32) / np.max(np.abs(clean_signal))
    enhanced_signal = enhanced_signal.astype(np.float32) / np.max(
        np.abs(enhanced_signal)
    )

    pesq_score = calculate_pesq(clean_signal, enhanced_signal, fs_clean)
    stoi_score = calculate_stoi(clean_signal, enhanced_signal, fs_clean)

    return pesq_score, stoi_score


if __name__ == "__main__":
    pesq_result, stoi_result = benchmark("data/clean.wav", "data/enhanced.wav")
    print(f"PESQ: {pesq_result:.2f}\nSTOI: {stoi_result:.2f}")
