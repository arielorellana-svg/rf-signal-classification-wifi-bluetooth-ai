# RF Signal Classification for WiFi and Bluetooth Coexistence

MATLAB project for classifying RF activity from spectrogram images generated from complex I/Q observations. The classifier distinguishes WiFi, Bluetooth, coexistence conditions, receiver-like noise, and unknown RF-like activity.

Two trained models are included:

1. A custom convolutional neural network trained from scratch.
2. A ResNet-18 transfer-learning model adapted to the six RF classes.

The complete workflow is reproducible from synthetic waveform generation through independent evaluation. Pretrained model files and result tables are included so the evaluation stages can also be checked without retraining.

## Classification classes

| Class | Description |
| --- | --- |
| `Bluetooth` | Bluetooth Low Energy activity with randomized frequency position and RF impairments. |
| `Noise` | Complex receiver-like noise with colored components, DC leakage, and weak spurious tones. |
| `Unknown` | Synthetic RF-like activity outside the target WiFi and Bluetooth classes. |
| `WiFi` | IEEE 802.11 non-HT waveform generated with MATLAB WLAN functions. |
| `WiFi_Bluetooth_Overlap` | WiFi and Bluetooth signals occupying overlapping or nearby spectral regions. |
| `WiFi_Bluetooth_Separated` | WiFi and Bluetooth signals present in the same observation and separated in frequency. |

Each 224 × 224 spectrogram receives one class label. The project performs image-level classification rather than pixel-level segmentation.

## Workflow

```text
Synthetic I/Q waveform generation
        ↓
Domain randomization and RF impairments
        ↓
224 × 224 spectrogram generation
        ↓
Custom CNN and ResNet-18 training
        ↓
Independent blind-test evaluation
        ↓
Independent receiver-like validation
        ↓
Accuracy, precision, recall, F1-score, and confusion matrices
```

## Datasets

All datasets used for the reported results are generated synthetically.

| Dataset | Classes | Samples per class | Total | Purpose |
| --- | ---: | ---: | ---: | --- |
| Training dataset | 6 | 3,000 | 18,000 | Training, validation, and internal testing |
| Independent blind test | 6 | 1,000 | 6,000 | Final evaluation on separately generated samples |
| Receiver-like validation | 6 | 600 | 3,600 | Robustness evaluation under acquisition-inspired effects |

Canonical output locations:

```text
data/spectrograms_v3_domain_randomized
data/blind_test_v2_final
data/receiver_like_validation/spectrograms
```

The receiver-like generator applies a separate impairment chain containing randomized SNR, programmed frequency displacement, fine oscillator mismatch, multipath, IQ imbalance, DC leakage, simulated receiver gain, clipping, and ADC quantization. Its metadata records the controlled generation parameters and uses repository-relative file paths.

## WiFi and Bluetooth coexistence synthesis

WiFi and Bluetooth waveforms are resampled to the common 40 MS/s observation rate, translated explicitly in complex baseband, normalized independently, combined with randomized relative power, and passed through the selected impairment chain.

The implementation does not require `comm.MultibandCombiner`; explicit complex frequency translation and baseband addition provide direct control over spectral overlap, spectral separation, and relative signal strength.

## Models

### Custom CNN baseline

```text
models/cnn_wifi_bluetooth_v3_domain_randomized.mat
```

The network is trained from scratch using grayscale spectrograms.

### ResNet-18 transfer learning

```text
models/resnet18_transfer_learning_wifi_bluetooth.mat
```

The grayscale spectrograms are converted to RGB during preprocessing, and the final ResNet-18 layers are adapted to the six RF classes.

## Reported results

| Model | Independent blind test | Receiver-like validation |
| --- | ---: | ---: |
| Custom CNN baseline | 92.65% | 84.36% |
| ResNet-18 transfer learning | 93.50% | 86.69% |

ResNet-18 improves the blind-test accuracy by 0.85 percentage points and the receiver-like validation accuracy by 2.33 percentage points relative to the custom CNN baseline.

### Receiver-like metrics: custom CNN

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Bluetooth | 0.8447 | 0.9517 | 0.8950 |
| Noise | 0.8978 | 0.8633 | 0.8802 |
| Unknown | 0.6641 | 0.8733 | 0.7545 |
| WiFi | 0.8393 | 0.7050 | 0.7663 |
| WiFi_Bluetooth_Overlap | 0.9056 | 0.7833 | 0.8400 |
| WiFi_Bluetooth_Separated | 0.9925 | 0.8850 | 0.9357 |

### Receiver-like metrics: ResNet-18

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Bluetooth | 0.8902 | 0.8783 | 0.8842 |
| Noise | 0.8806 | 0.8850 | 0.8828 |
| Unknown | 0.7229 | 0.8567 | 0.7841 |
| WiFi | 0.8402 | 0.8850 | 0.8620 |
| WiFi_Bluetooth_Overlap | 0.9370 | 0.8183 | 0.8737 |
| WiFi_Bluetooth_Separated | 0.9796 | 0.8783 | 0.9262 |

Detailed CSV files are available in `results/`.

## MATLAB requirements

- MATLAB
- Deep Learning Toolbox
- WLAN Toolbox
- Bluetooth Toolbox
- Communications Toolbox
- Signal Processing Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox Model for ResNet-18 Network, for transfer learning

## Reproduce the project

Open MATLAB in the repository root. The generation and training scripts write to canonical project-relative folders.

### 1. Generate the training dataset

```matlab
run("matlab/step01_generate_dataset_wifi_bluetooth.m")
```

### 2. Train the models

```matlab
run("matlab/step02_train_cnn_wifi_bluetooth.m")
run("matlab/step02b_train_transfer_learning_wifi_bluetooth.m")
```

### 3. Generate independent evaluation datasets

```matlab
run("matlab/step03_generate_blind_test.m")
run("matlab/step04_generate_receiver_like_validation.m")
```

### 4. Evaluate both models

```matlab
run("matlab/step07_run_all_evaluations.m")
```

Individual evaluation scripts are also available:

```matlab
run("matlab/step05_evaluate_blind_test.m")
run("matlab/step05b_evaluate_transfer_learning_blind_test.m")
run("matlab/step06_evaluate_cnn_receiver_like.m")
run("matlab/step06b_evaluate_transfer_learning_receiver_like.m")
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for expected outputs, random seeds, dataset compatibility paths, and verification steps.

## Repository structure

```text
rf-signal-classification-wifi-bluetooth-ai/
├── README.md
├── REPRODUCIBILITY.md
├── LICENSE
├── matlab/
│   ├── step00_test_wifi_bluetooth_generation.m
│   ├── step01_generate_dataset_wifi_bluetooth.m
│   ├── step02_train_cnn_wifi_bluetooth.m
│   ├── step02b_train_transfer_learning_wifi_bluetooth.m
│   ├── step03_generate_blind_test.m
│   ├── step04_generate_receiver_like_validation.m
│   ├── step05_evaluate_blind_test.m
│   ├── step05b_evaluate_transfer_learning_blind_test.m
│   ├── step06_evaluate_cnn_receiver_like.m
│   ├── step06b_evaluate_transfer_learning_receiver_like.m
│   ├── step07_run_all_evaluations.m
│   └── step11_usrp_b210_real_capture.m
├── models/
│   ├── cnn_wifi_bluetooth_v3_domain_randomized.mat
│   ├── resnet18_transfer_learning_wifi_bluetooth.mat
│   └── README.md
├── data/
│   └── README.md
└── results/
    ├── README.md
    ├── metrics_*.csv
    ├── summary_*.csv
    └── confusion_matrix_*.png
```

## Notes

- The blind-test and receiver-like datasets are generated independently from the training set.
- Random seeds and dataset configurations are recorded to support repeatable generation.
- The `Unknown` class is intentionally broad and may share visual characteristics with noise, bursts, or weak narrowband activity.
- An optional USRP B210 capture utility is retained for later hardware experiments but is not required to reproduce the reported results.

## License

See [LICENSE](LICENSE).
