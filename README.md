# BigVSAN [ICASSP 2024]

This repository contains the official PyTorch implementation of **"BigVSAN: Enhancing GAN-based Neural Vocoders with Slicing Adversarial Network"** (*[arXiv 2309.02836](https://arxiv.org/abs/2309.02836)*).
Please cite [[1](#citation)] in your work when using this code in your experiments.

### [Audio demo](https://takashishibuyasony.github.io/bigvsan/)

## Installation
This repository builds on the codebase of [BigVGAN](https://github.com/NVIDIA/BigVGAN).

Download the LibriTTS dataset [here](http://www.openslr.org/60/) in advance.

Clone the repository and install dependencies.
```shell
# the codebase has been tested on Python 3.8 with PyTorch 1.13.0
git clone https://github.com/sony/bigvsan
pip install -r requirements.txt
```

Create symbolic link to the root of the dataset. The codebase uses filelist with the relative path from the dataset. Below are the example commands for LibriTTS dataset.
``` shell
cd LibriTTS && \
ln -s /path/to/your/LibriTTS/train-clean-100 train-clean-100 && \
ln -s /path/to/your/LibriTTS/train-clean-360 train-clean-360 && \
ln -s /path/to/your/LibriTTS/train-other-500 train-other-500 && \
ln -s /path/to/your/LibriTTS/dev-clean dev-clean && \
ln -s /path/to/your/LibriTTS/dev-other dev-other && \
ln -s /path/to/your/LibriTTS/test-clean test-clean && \
ln -s /path/to/your/LibriTTS/test-other test-other && \
cd ..
```

## Training
Train BigVSAN model. Below is an example command for training BigVSAN using LibriTTS dataset at 24kHz with a full 100-band mel spectrogram as input.
```shell
python train.py \
--config configs/bigvsan_24khz_100band.json \
--input_wavs_dir LibriTTS \
--input_training_file LibriTTS/train-full.txt \
--input_validation_file LibriTTS/val-full.txt \
--list_input_unseen_wavs_dir LibriTTS LibriTTS \
--list_input_unseen_validation_file LibriTTS/dev-clean.txt LibriTTS/dev-other.txt \
--checkpoint_path exp/bigvsan
```

## Evaluation
We evaluated our BigVSAN model as follows:

Generate and save audio samples after you finish model training. Below is an example command for generating and save audio samples for evaluation.
```shell
python train.py \
--config configs/bigvsan_24khz_100band.json \
--input_wavs_dir LibriTTS \
--input_training_file LibriTTS/train-full.txt \
--input_validation_file LibriTTS/val-full.txt \
--list_input_unseen_wavs_dir LibriTTS LibriTTS \
--list_input_unseen_validation_file LibriTTS/dev-clean.txt LibriTTS/dev-other.txt \
--checkpoint_path exp/bigvsan \
--evaluate True \
--eval_subsample 1 \
--skip_seen True \
--save_audio True
```

Run the evaluation tool provided [here](https://github.com/sony/bigvsan_eval). It computes five objective metric scores: M-STFT, PESQ, MCD, Periodicity, and V/UV F1.
```shell
python evaluate.py \
../bigvsan/exp/bigvsan/samples/gt_unseen_LibriTTS-dev-clean ../bigvsan/exp/bigvsan/samples/unseen_LibriTTS-dev-clean_01000001 \
../bigvsan/exp/bigvsan/samples/gt_unseen_LibriTTS-dev-other ../bigvsan/exp/bigvsan/samples/unseen_LibriTTS-dev-other_01000001
```
It will take about an hour to complete an evaluation. Note that, when audio samples are generated and saved with `train.py`, it also outputs M-STFT and PESQ scores, but their values will be different from the output of `evaluate.py`. This is due to 16-bit quantization for saving a sample as a wav file.


## Synthesis
Synthesize from BigVSAN model. Below is an example command for generating audio from the model.
It computes mel spectrograms using wav files from `--input_wavs_dir` and saves the generated audio to `--output_dir`.
```shell
python inference.py \
--checkpoint_file exp/bigvsan/g_01000000 \
--input_wavs_dir /path/to/your/input_wav \
--output_dir /path/to/your/output_wav
```

## Pretrained Models
We provide pretrained checkpoints trained on the LibriTTS dataset [here](https://zenodo.org/records/10037439).
You can download zip files, each of which contains checkpoints of a generator (e.g., g_01000000) and a discriminator (e.g., do_01000000).

|Zip file name|# of training steps|M-STFT|PESQ|MCD|Periodicity|V/UV F1|
|---|---|---|---|---|---|---|
|`bigvsan_01mstep`|1,000,000|0.7881|4.116|0.3381|0.0935|0.9635|
|`bigvsan_10mstep`|10,000,000|0.7210|4.316|0.3065|0.0726|0.9729|

The paper results are based on `bigvsan_01mstep`.

## Citation
[1] Shibuya, T., Takida, Y., Mitsufuji, Y.,
"BigVSAN: Enhancing GAN-based Neural Vocoders with Slicing Adversarial Network,"
ICASSP 2024.
```bibtex
@inproceedings{shibuya2024bigvsan,
        title={{BigVSAN}: Enhancing GAN-based Neural Vocoders with Slicing Adversarial Network},
        author={Shibuya, Takashi and Takida, Yuhta and Mitsufuji, Yuki},
        booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        year={2024}
}
```

## References
* [BigVGAN](https://github.com/NVIDIA/BigVGAN)

* [HiFi-GAN](https://github.com/jik876/hifi-gan) (for generator and multi-period discriminator)

* [Snake](https://github.com/EdwardDixon/snake) (for periodic activation)

* [Alias-free-torch](https://github.com/junjun3518/alias-free-torch) (for anti-aliasing)

* [Julius](https://github.com/adefossez/julius) (for low-pass filter)

* [UnivNet](https://github.com/mindslab-ai/univnet) (for multi-resolution discriminator)