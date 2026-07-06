# RF Signal Classification Using AI: WiFi and Bluetooth Coexistence

This project implements an AI-based RF signal classifier for WiFi, Bluetooth, noise, unknown signals, and WiFi-Bluetooth coexistence scenarios using MATLAB, spectrograms, and convolutional neural networks.

The project is inspired by the MathWorks / National Instruments challenge project: **Classify RF Signals Using AI**.

## Objective

Develop a deep learning system capable of classifying RF signals from time-frequency spectrograms generated from synthetic I/Q waveforms and, later, real SDR captures.

## Classes

The final CNN model classifies six classes:

- Bluetooth
- Noise
- Unknown
- WiFi
- WiFi_Bluetooth_Overlap
- WiFi_Bluetooth_Separated

## Methodology

1. Generate WiFi and Bluetooth I/Q waveforms in MATLAB.
2. Apply RF impairments such as AWGN, frequency offset, phase offset, multipath, IQ imbalance, DC offset, and amplitude variation.
3. Convert I/Q waveforms into normalized spectrogram images.
4. Train a CNN classifier using a domain-randomized synthetic dataset.
5. Evaluate the model using independent blind test datasets.
6. Prepare validation with public NIST I/Q recordings and future USRP B210 captures.

## Dataset

The main synthetic training dataset contains:

- 18,000 spectrogram images.
- 6 classes.
- 3,000 images per class.

The final independent test dataset contains:

- 6,000 spectrogram images.
- 6 classes.
- 1,000 images per class.

Large datasets are not included in this repository. They can be regenerated using the MATLAB scripts in the `matlab/` folder.

## Final Results

| Test | Accuracy |
|---|---:|
| Internal V3 test | 95.52% |
| Blind test V1 | 94.50% |
| Final independent blind test | 92.65% |

## Final Independent Test Metrics

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bluetooth | 0.9115 | 0.9270 | 0.9192 |
| Noise | 0.9861 | 0.9950 | 0.9905 |
| Unknown | 0.9969 | 0.9520 | 0.9739 |
| WiFi | 0.8104 | 0.9790 | 0.8868 |
| WiFi_Bluetooth_Overlap | 0.9437 | 0.7880 | 0.8589 |
| WiFi_Bluetooth_Separated | 0.9406 | 0.9180 | 0.9292 |

## Requirements

Tested with MATLAB R2025b.

Required toolboxes:

- WLAN Toolbox
- Bluetooth Toolbox
- Communications Toolbox
- Signal Processing Toolbox
- Deep Learning Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox

Optional for SDR validation:

- Communications Toolbox Support Package for USRP Radio
- Wireless Testbench

## How to Run

Run the scripts from the `matlab/` folder in this order:

1. `01_generate_dataset_wifi_bluetooth_v3.m`
2. `02_train_cnn_wifi_bluetooth_v3.m`
3. `03_generate_blind_test_v2_final.m`
4. `05_evaluate_blind_test_v2_final.m`

## Current Status

The synthetic dataset generation, CNN training, and independent blind test evaluation have been completed.

Public NIST I/Q recordings and future USRP B210 captures will be used for real-world validation.

## Future Work

- Validate the classifier with public NIST WiFi/Bluetooth I/Q recordings.
- Capture real 2.4 GHz signals using USRP B210.
- Compare the custom CNN with transfer learning models such as ResNet-18 and MobileNetV2.
- Extend the system to semantic segmentation for time-frequency localization.
- Apply pruning or quantization for faster inference.

## Author

Ariel Orellana
