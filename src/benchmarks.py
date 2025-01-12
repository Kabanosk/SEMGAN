import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate enhanced audio files")
    parser.add_argument(
        "-e",
        "--enhanced_dir",
        type=str,
        required=True,
        help="Directory containing enhanced audio files (with enhanced_ prefix)",
    )
    parser.add_argument(
        "-c",
        "--clean_dir",
        type=str,
        required=True,
        help="Directory containing original clean audio files",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file",
    )
    return parser.parse_args()


def calculate_metrics(clean_path, enhanced_path, sample_rate):
    try:
        # Load audio files
        clean, sr = torchaudio.load(str(clean_path))
        enhanced, esr = torchaudio.load(str(enhanced_path))

        # Convert to mono first
        if clean.size(0) > 1:
            clean = torch.mean(clean, dim=0, keepdim=True)
        if enhanced.size(0) > 1:
            enhanced = torch.mean(enhanced, dim=0, keepdim=True)

        # Resample if needed
        if sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            clean = resampler(clean)
        if esr != sample_rate:
            resampler = T.Resample(esr, sample_rate)
            enhanced = resampler(enhanced)

        # Ensure same length
        min_len = min(clean.size(-1), enhanced.size(-1))
        clean = clean[..., :min_len]
        enhanced = enhanced[..., :min_len]

        # Prepare for PESQ and STOI (need to be mono and correct shape)
        clean_np = clean.squeeze().numpy()
        enhanced_np = enhanced.squeeze().numpy()

        # Calculate metrics
        try:
            pesq_score = pesq(sample_rate, clean_np, enhanced_np, "wb")
        except Exception as e:
            print(f"PESQ calculation failed for {enhanced_path.name}: {str(e)}")
            pesq_score = float("nan")

        try:
            stoi_score = stoi(clean_np, enhanced_np, sample_rate, extended=False)
        except Exception as e:
            print(f"STOI calculation failed for {enhanced_path.name}: {str(e)}")
            stoi_score = float("nan")

        return {"pesq": pesq_score, "stoi": stoi_score}

    except Exception as e:
        print(f"Error processing file pair:")
        print(f"Clean: {clean_path}")
        print(f"Enhanced: {enhanced_path}")
        print(f"Error: {str(e)}")
        return {"pesq": float("nan"), "stoi": float("nan")}


def get_statistics(scores):
    if not scores or len(scores) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)) if len(scores) > 1 else 0.0,
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "count": len(scores),
    }


def evaluate_enhanced_files(args):
    enhanced_dir = Path(args.enhanced_dir)
    clean_dir = Path(args.clean_dir)

    if not enhanced_dir.exists():
        raise ValueError(f"Enhanced directory not found: {enhanced_dir}")
    if not clean_dir.exists():
        raise ValueError(f"Clean directory not found: {clean_dir}")

    enhanced_files = list(enhanced_dir.glob("enhanced_*.wav"))
    if not enhanced_files:
        raise ValueError(f"No enhanced files found in {enhanced_dir}")

    print(f"Found {len(enhanced_files)} enhanced files")

    results = []
    successful_evaluations = 0

    for enhanced_file in tqdm(enhanced_files, desc="Evaluating files"):
        # Remove 'enhanced_' prefix to find corresponding clean file
        clean_filename = enhanced_file.name.replace("enhanced_", "")
        clean_file = clean_dir / clean_filename

        if not clean_file.exists():
            print(f"Warning: No matching clean file found for {enhanced_file.name}")
            continue

        metrics = calculate_metrics(clean_file, enhanced_file, args.sample_rate)
        results.append(
            {"file": enhanced_file.name, "clean_file": clean_filename, **metrics}
        )

        if not (np.isnan(metrics["pesq"]) and np.isnan(metrics["stoi"])):
            successful_evaluations += 1

    # Calculate statistics for valid scores only
    pesq_scores = [r["pesq"] for r in results if not np.isnan(r["pesq"])]
    stoi_scores = [r["stoi"] for r in results if not np.isnan(r["stoi"])]

    summary = {
        "evaluation_summary": {
            "total_files": len(enhanced_files),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": len(enhanced_files) - successful_evaluations,
        },
        "metrics": {
            "pesq": get_statistics(pesq_scores),
            "stoi": get_statistics(stoi_scores),
        },
        "individual_results": results,
    }

    # Save results
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)

    return summary


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_enhanced_files(args)

    print("\nEvaluation Summary:")
    print(f"Total files: {results['evaluation_summary']['total_files']}")
    print(
        f"Successful evaluations: {results['evaluation_summary']['successful_evaluations']}"
    )
    print(f"Failed evaluations: {results['evaluation_summary']['failed_evaluations']}")

    print("\nPESQ Scores:")
    pesq_metrics = results["metrics"]["pesq"]
    if pesq_metrics["count"] > 0:
        print(f"Number of valid scores: {pesq_metrics['count']}")
        print(f"Mean: {pesq_metrics['mean']:.3f}")
        print(f"Std:  {pesq_metrics['std']:.3f}")
        print(f"Min:  {pesq_metrics['min']:.3f}")
        print(f"Max:  {pesq_metrics['max']:.3f}")
    else:
        print("No valid PESQ scores calculated")

    print("\nSTOI Scores:")
    stoi_metrics = results["metrics"]["stoi"]
    if stoi_metrics["count"] > 0:
        print(f"Number of valid scores: {stoi_metrics['count']}")
        print(f"Mean: {stoi_metrics['mean']:.3f}")
        print(f"Std:  {stoi_metrics['std']:.3f}")
        print(f"Min:  {stoi_metrics['min']:.3f}")
        print(f"Max:  {stoi_metrics['max']:.3f}")
    else:
        print("No valid STOI scores calculated")
