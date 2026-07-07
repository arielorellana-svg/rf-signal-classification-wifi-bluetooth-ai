$repo = "C:\Users\DETPC\Documents\GitHub\rf-signal-classification-wifi-bluetooth-ai"

if (!(Test-Path "$repo\.git")) {
    Write-Error "No se encontró .git en el repositorio. Revisa la ruta."
    exit
}

New-Item -ItemType Directory -Force -Path "$repo\docs" | Out-Null

@'
# RF Signal Classification Using AI: WiFi and Bluetooth Coexistence

This repository presents an AI-based RF signal classification system for WiFi, Bluetooth, noise, unknown signals, and WiFi-Bluetooth coexistence scenarios.

The system was developed in MATLAB using synthetic I/Q waveform generation, RF impairment modeling, spectrogram-based feature extraction, and convolutional neural network classification.

The project follows the technical direction of the MathWorks / National Instruments challenge project **Classify RF Signals Using AI**, which focuses on using deep learning to classify wireless signals and validate the approach with software-defined radio workflows.

## Project Overview

The classifier identifies six RF signal scenarios:

- Bluetooth
- Noise
- Unknown
- WiFi
- WiFi_Bluetooth_Overlap
- WiFi_Bluetooth_Separated

The model does not use RSSI alone. Instead, I/Q waveforms are converted into normalized time-frequency spectrograms. These spectrograms are used as image-like inputs to a CNN classifier.

## Signal Processing Workflow

```text
Synthetic or captured I/Q waveform
        ↓
RF impairments and channel effects
        ↓
Time-frequency spectrogram
        ↓
224 × 224 normalized image
        ↓
CNN classifier
        ↓
Predicted RF signal class
---

## Simulated SDR Validation

A simulated SDR validation dataset was generated to evaluate the trained CNN under receiver-like conditions. This dataset includes gain variation, ADC-like quantization, IQ imbalance, DC offset, frequency offset, multipath, colored noise, and burst-like signal activity.

| Validation stage | Dataset size | Accuracy |
|---|---:|---:|
| Final independent blind test | 6,000 samples | 92.65% |
| Simulated SDR validation | 3,600 samples | 84.36% |

### Simulated SDR Validation Metrics

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Bluetooth | 0.8447 | 0.9517 | 0.8950 |
| Noise | 0.8978 | 0.8633 | 0.8802 |
| Unknown | 0.6641 | 0.8733 | 0.7545 |
| WiFi | 0.8393 | 0.7050 | 0.7663 |
| WiFi_Bluetooth_Overlap | 0.9056 | 0.7833 | 0.8400 |
| WiFi_Bluetooth_Separated | 0.9925 | 0.8850 | 0.9357 |

### Simulated SDR Confusion Matrix

![Simulated SDR confusion matrix](results/confusion_matrix_simulated_sdr_dataset.png)
