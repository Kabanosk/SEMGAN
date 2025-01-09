# SEMGAN

SEMGAN is a Speech Enhancement Multi-Discriminator Generative Adversarial Network implemented in Python. This project was developed as part of my B.Eng. thesis.

## Prerequisites

- Python 3.11 or higher

## Installation

1. Clone the repository:
   ```shell
   git clone git@github.com:Kabanosk/SEMGAN.git
   ```

2. Navigate to the project directory:
   ```shell
   cd semgan
   ```

3. Install the dependencies using Poetry:
   ```shell
   poetry install
   ```

## Usage

### Training
To train the SEMGAN model, run the following command:
```shell
PYTHONPATH="." poetry run python3 src/train.py -c src/config/config.yaml
```

Make sure to update the configuration file `src/config/config.yaml` with the desired settings before running the training script.

### Inferance
To enhance audio files using a trained SEMGAN model, use the following command:
```sh
PYTHONPATH="." poetry run python3 src/infer.py \ 
    --model semgan \
    --input path/to/input/audio \
    --output path/to/output/directory \
    --checkpoint path/to/model/checkpoint
```
Parameters:

- `-m`, `--model`: Model architecture (segan or semgan)
- `-i`, `--input`: Path to input audio file or directory containing WAV files
- `-o`, `--output`: Path to output directory for enhanced audio
- `--checkpoint`: Path to trained model checkpoint
- `--sample_rate`: Target sample rate (default: 16000)
- `--segment_length`: Audio segment length for processing (default: 16384)

The script will process all WAV files in the input directory and save the enhanced versions in the output directory.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For any questions or inquiries, please contact Wojciech Fiołka at fiolkawojciech@gmail.com.