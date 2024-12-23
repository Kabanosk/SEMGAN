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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For any questions or inquiries, please contact Wojciech Fiołka at fiolkawojciech@gmail.com.