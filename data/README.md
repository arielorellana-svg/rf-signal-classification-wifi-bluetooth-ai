# Data

This project uses three synthetic datasets for training and evaluation.

## Training dataset

```text
data/spectrograms_v3_domain_randomized
```

- 6 classes.
- 3,000 spectrograms per class.
- 18,000 images in total.
- Used for training, validation, and internal testing.

## Independent blind test

```text
data/blind_test_v2_final
```

- 6 classes.
- 1,000 spectrograms per class.
- 6,000 images in total.
- Generated independently and not used during training.

Some repository copies may store this dataset under `results/blindtestv2_final` or `results/blind_test_v2_final`. The evaluation scripts detect these layouts automatically.

## Receiver-like validation dataset

The additional 3,600-image validation set is synthetic. It contains 600 samples per class and applies receiver-inspired effects such as randomized SNR, programmed frequency offsets, simulated gain variation, IQ imbalance, DC offset, channel effects, colored noise, bursts, and weak spurious components.

Known SNR, gain, and offset values in its metadata are simulation parameters. They are not measurements from RF hardware.

Supported locations are:

```text
results/data_sdr/spectrograms
results/data_sdr
data_simulated_sdr_v1/spectrograms
data_simulated_sdr_v1
data_sdr/spectrograms
data_sdr
```

The first location containing all six class folders is selected automatically.

Some legacy filenames use `sdr` or `sdrsim`. In this repository, those names refer to the synthetic SDR-like or receiver-like validation set and do not indicate hardware capture.

## Classes

```text
Bluetooth
Noise
Unknown
WiFi
WiFi_Bluetooth_Overlap
WiFi_Bluetooth_Separated
```

## Reproducibility

The synthetic training and blind-test datasets can be regenerated with the MATLAB scripts in the `matlab/` folder. The pretrained models in `models/` allow the included evaluation results to be checked without retraining.
