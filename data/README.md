# Data

Large datasets are not included in this repository because of size.

## Synthetic datasets

The following datasets can be regenerated using the MATLAB scripts in the `matlab/` folder:

- `data/spectrograms_v3_domain_randomized`: 18,000 synthetic spectrograms for training, validation, and internal testing.
- `data/blind_test_v2_final`: 6,000 independent blind-test spectrograms.

## Real SDR validation dataset

The SDR validation dataset was generated from real USRP B210 I/Q captures:

- `data_sdr/spectrograms`: 3,600 real SDR spectrogram images used for additional validation.

This dataset is not simulated. Recreating it requires compatible SDR hardware and the Communications Toolbox Support Package for USRP Radio.
