# RF Signal Classification Using AI: WiFi and Bluetooth Coexistence

This repository presents a MATLAB-based deep learning system for classifying RF signal conditions in WiFi and Bluetooth coexistence scenarios. The project converts complex I/Q waveforms into time-frequency spectrogram images and uses convolutional neural networks to classify the corresponding RF scenario.

The project includes two model workflows:

1. A custom CNN baseline trained from scratch.
2. A ResNet-18 transfer-learning model adapted from a pretrained image-classification network.

The final system classifies six RF signal classes:

- `Bluetooth`
- `Noise`
- `Unknown`
- `WiFi`
- `WiFi_Bluetooth_Overlap`
- `WiFi_Bluetooth_Separated`

---

## 1. Project Objective

The objective of this project is to develop and evaluate an AI-based RF signal classifier capable of distinguishing WiFi, Bluetooth, noise, unknown RF-like signals, and WiFi-Bluetooth coexistence cases.

The project focuses on spectrum-sensing classification using spectrograms generated from I/Q waveforms. Instead of relying only on received power or RSSI, the classifier learns time-frequency patterns such as occupied bandwidth, spectral location, frequency displacement, overlap, coexistence behavior, and noise-like structures.

---

## 2. Technical Approach

The processing chain is organized as follows:

1. Generate complex baseband I/Q waveforms for WiFi, Bluetooth, noise, unknown signals, and coexistence cases.
2. Apply domain randomization, RF impairments, and channel-like effects.
3. Convert I/Q waveform segments into normalized spectrogram images.
4. Train a custom CNN baseline.
5. Train a ResNet-18 transfer-learning model.
6. Evaluate both models on an independent blind-test dataset.
7. Evaluate both models on SDR spectrogram images.

The repository is designed so that the main results can be reproduced from MATLAB scripts and verified using saved model files and result summaries.

---

## 3. Signal Classes

| Class | Description |
|---|---|
| `Bluetooth` | Bluetooth Low Energy waveform with frequency displacement and RF impairments. |
| `Noise` | Complex receiver-like noise, including colored noise, DC offset, and weak artifacts. |
| `Unknown` | Synthetic RF-like signals that do not belong to the WiFi or Bluetooth target classes. |
| `WiFi` | WLAN waveform generated using MATLAB WLAN functions. |
| `WiFi_Bluetooth_Overlap` | WiFi and Bluetooth activity occupying overlapping or nearby spectral regions. |
| `WiFi_Bluetooth_Separated` | WiFi and Bluetooth activity present in the same observation window but separated in frequency. |

---

## 4. Dataset Summary

Large generated datasets are not included in this repository because of size. They can be regenerated using the MATLAB scripts in the `matlab/` folder.

| Dataset | Classes | Samples per class | Total samples | Purpose |
|---|---:|---:|---:|---|
| `data/spectrograms_v3_domain_randomized` | 6 | 3,000 | 18,000 | Training, validation, and internal testing |
| `data/blind_test_v2_final` | 6 | 1,000 | 6,000 | Final independent blind-test evaluation |
| `data_sdr/spectrograms` | 6 | 600 | 3,600 | SDR spectrogram image validation |

The training dataset and final blind-test dataset are generated independently using different random seeds and parameter variations. The SDR dataset is used as an additional validation stage to evaluate model behavior under captured SDR image conditions.

---

## 5. RF Impairments and Domain Randomization

To improve generalization, the synthetic waveform generation process includes randomized RF and receiver-like effects, including:

- Additive white Gaussian noise
- Frequency offset
- Random phase offset
- Amplitude variation
- Multipath channel effects
- IQ imbalance
- DC offset
- Colored noise
- Variable Bluetooth-to-WiFi power ratio
- Continuous Bluetooth and unknown-signal frequency displacement
- Time shifts
- Noise bursts and weak spurious tones

This variability reduces dependence on idealized signal positions and encourages the model to learn spectral structure instead of memorizing fixed locations.

---

## 6. Spectrogram Representation

Each I/Q segment is converted into a normalized spectrogram image with fixed size:

```text
224 x 224
