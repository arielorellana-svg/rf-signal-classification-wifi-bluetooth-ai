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