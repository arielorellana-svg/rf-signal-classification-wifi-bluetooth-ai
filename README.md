# RF Signal Classification Using AI: WiFi and Bluetooth Coexistence

This repository presents a MATLAB-based deep learning system for RF signal classification in WiFi and Bluetooth coexistence scenarios. The project focuses on classifying wireless signals from time-frequency representations generated from complex I/Q waveforms.

The implemented classifier identifies six signal conditions:

- Bluetooth
- Noise
- Unknown
- WiFi
- WiFi_Bluetooth_Overlap
- WiFi_Bluetooth_Separated

The system follows the technical objective of the MathWorks / National Instruments challenge project **Classify RF Signals Using AI**, which proposes using deep learning to classify wireless signals and evaluate RF coexistence scenarios.

---

## 1. Project Objective

The objective of this project is to develop an AI-based RF signal classifier capable of distinguishing WiFi, Bluetooth, noise, unknown signals, and WiFi-Bluetooth coexistence cases using spectrogram-based deep learning.

The project addresses the RF spectrum sensing problem by converting complex I/Q waveforms into time-frequency spectrograms and using a convolutional neural network to classify the corresponding RF scenario.

---

## 2. Technical Approach

The classification system is based on the following processing chain:

1. Generation of complex baseband I/Q waveforms.
2. Construction of WiFi, Bluetooth, noise, unknown, and coexistence scenarios.
3. Application of RF impairments and channel effects.
4. Conversion of I/Q samples into normalized spectrogram images.
5. Training of a convolutional neural network.
6. Independent validation using blind-test datasets.

The model does not rely on RSSI or average received power as the main feature. Instead, it uses time-frequency information extracted from I/Q signals. This allows the classifier to learn spectral occupancy, bandwidth, frequency displacement, coexistence behavior, and interference patterns.

---

## 3. Signal Classes

| Class | Description |
|---|---|
| Bluetooth | Bluetooth Low Energy waveform with frequency displacement and RF impairments. |
| Noise | Complex noise conditions, including colored noise and receiver-like artifacts. |
| Unknown | Synthetic RF-like signals not belonging to the target WiFi/Bluetooth classes. |
| WiFi | WLAN waveform generated using MATLAB WLAN functions. |
| WiFi_Bluetooth_Overlap | WiFi and Bluetooth signals occupying overlapping frequency regions. |
| WiFi_Bluetooth_Separated | WiFi and Bluetooth signals present in the same observation window but separated in frequency. |

---

## 4. Dataset Generation

The dataset was generated in MATLAB using synthetic I/Q waveforms. WiFi waveforms were generated using WLAN configurations, while Bluetooth waveforms were generated using Bluetooth Low Energy waveform generation. The generated signals were translated in frequency, combined when necessary, impaired, and converted into spectrograms.

The final training dataset contains:

| Dataset | Classes | Samples per class | Total samples |
|---|---:|---:|---:|
| Domain-randomized training dataset | 6 | 3,000 | 18,000 |
| Final independent blind-test dataset | 6 | 1,000 | 6,000 |

Large datasets are not included in this repository. They can be regenerated using the MATLAB scripts provided in the `matlab/` folder.

---

## 5. RF Impairments and Variability

To improve generalization, the dataset includes domain-randomized RF conditions. The following effects were included during waveform generation:

- Additive white Gaussian noise
- Frequency offset
- Phase offset
- Amplitude variation
- Multipath channel effects
- IQ imbalance
- DC offset
- Colored noise
- Variable Bluetooth-to-WiFi power ratio
- Continuous frequency displacement
- Unknown RF-like signal patterns

This process creates a more diverse dataset and reduces dependence on idealized waveform conditions.

---

## 6. Spectrogram Representation

Each I/Q waveform segment is converted into a spectrogram and resized to a fixed image size of `224 × 224`.

The spectrogram representation preserves relevant time-frequency characteristics such as:

- Occupied bandwidth
- Frequency displacement
- Signal coexistence
- Spectral overlap
- Noise-like behavior
- Unknown signal structures

These spectrograms are used as the input to the CNN classifier.

---

## 7. CNN Classifier

A custom convolutional neural network was trained using the generated spectrogram dataset. The network was designed for image-based RF classification and trained with six output classes.

The model was trained and evaluated in MATLAB using the Deep Learning Toolbox.

## Pretrained Model

A pretrained MATLAB CNN model is included in this repository at:

`models/cnn_wifi_bluetooth_v3_domain_randomized.mat`

This file allows reviewers to verify the reported blind-test results without retraining the network from scratch.

To evaluate the pretrained model, run the following script from the repository root in MATLAB:

```matlab
run("matlab/step05_evaluate_blind_test.m")
```

If the model needs to be regenerated, run:

```matlab
run("matlab/step01_generate_dataset_wifi_bluetooth.m")
run("matlab/step02_train_cnn_wifi_bluetooth.m")
```

The evaluation script loads the pretrained model from `models/cnn_wifi_bluetooth_v3_domain_randomized.mat` and evaluates it on `data/blind_test_v2_final`.

---


## 8. Validation Methodology

The validation process was performed in three stages:

| Stage | Description |
|---|---|
| Internal V3 test | Evaluation using a held-out portion of the domain-randomized dataset. |
| Blind test V1 | Evaluation using a separately generated blind-test dataset. |
| Final independent blind test | Final evaluation using an independently generated dataset with 6,000 spectrograms. |

The final independent blind test was not used during training.

---

## 9. Results

The final CNN model achieved the following classification performance:

| Evaluation stage | Accuracy |
|---|---:|
| Internal V3 test | 95.52% |
| Blind test V1 | 94.50% |
| Final independent blind test | 92.65% |

---

## 10. Final Independent Test Metrics

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bluetooth | 0.9115 | 0.9270 | 0.9192 |
| Noise | 0.9861 | 0.9950 | 0.9905 |
| Unknown | 0.9969 | 0.9520 | 0.9739 |
| WiFi | 0.8104 | 0.9790 | 0.8868 |
| WiFi_Bluetooth_Overlap | 0.9437 | 0.7880 | 0.8589 |
| WiFi_Bluetooth_Separated | 0.9406 | 0.9180 | 0.9292 |

The final independent test reached an overall accuracy of **92.65%**.

---

## 11. Final Independent Confusion Matrix

The following confusion matrix summarizes the final independent evaluation:

![Final independent confusion matrix](results/confusion_matrix_blind_test_v3_final.png)

---

## 12. Repository Structure

```text
rf-signal-classification-wifi-bluetooth-ai/
├── README.md
├── .gitignore
├── matlab/
│   ├── step00_test_wifi_bluetooth_generation.m
│   ├── step01_generate_dataset_wifi_bluetooth.m
│   ├── step02_train_cnn_wifi_bluetooth.m
│   ├── step03_generate_blind_test.m
│   ├── step05_evaluate_blind_test.m
│   └── step11_usrp_b210_real_capture.m
├── results/
│   ├── confusion_matrix_blind_test_v3_final.png
│   └── metrics_blind_test_v3_final.csv
├── data/
│   └── README.md
├── models/
│   └── README.md
└── docs/
    └── validation_summary.md
```
---

## USRP B210 Validation

| Validation stage | Dataset size | Accuracy |
|---|---:|---:|
| Final independent blind test | 6,000 samples | 92.65% |
| USRP B210 validation | 3,600 samples | 84.36% |

### USRP B210 Validation Metrics

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bluetooth | 0.8447 | 0.9517 | 0.8950 |
| Noise | 0.8978 | 0.8633 | 0.8802 |
| Unknown | 0.6641 | 0.8733 | 0.7545 |
| WiFi | 0.8393 | 0.7050 | 0.7663 |
| WiFi_Bluetooth_Overlap | 0.9056 | 0.7833 | 0.8400 |
| WiFi_Bluetooth_Separated | 0.9925 | 0.8850 | 0.9357 |

### USRP B210 Confusion Matrix

![Real USRP B210 confusion matrix](results/confusion_matrix_simulated_sdr_dataset.png)

