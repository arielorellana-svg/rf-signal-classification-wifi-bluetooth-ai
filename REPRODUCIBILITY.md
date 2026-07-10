# Reproducibility Guide

This guide describes how to verify the included trained models and how to regenerate the complete synthetic workflow.

## 1. Environment

Recommended MATLAB products:

- Deep Learning Toolbox
- WLAN Toolbox
- Bluetooth Toolbox
- Communications Toolbox
- Signal Processing Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Deep Learning Toolbox Model for ResNet-18 Network

The scripts use local functions in script files and should be run with a MATLAB release that supports this feature.

## 2. Canonical project paths

```text
data/spectrograms_v3_domain_randomized
data/blind_test_v2_final
data/receiver_like_validation/spectrograms
models/
results/
```

Evaluation scripts determine the repository root from their own file location. They do not require a particular active MATLAB folder.

For compatibility with previously generated copies, the evaluators can also detect:

```text
results/blindtestv2_final
results/blind_test_v2_final
results/data_sdr/spectrograms
results/data_sdr
data_simulated_sdr_v1/spectrograms
data_simulated_sdr_v1
```

New dataset generation always uses the canonical `data/` locations.

## 3. Dataset definitions

| Dataset | Seed | Samples per class | Total samples |
| --- | ---: | ---: | ---: |
| Training dataset | 21 | 3,000 | 18,000 |
| Independent blind test | 2026 | 1,000 | 6,000 |
| Receiver-like validation | 407 | 600 | 3,600 |

All three datasets contain these six folders:

```text
Bluetooth
Noise
Unknown
WiFi
WiFi_Bluetooth_Overlap
WiFi_Bluetooth_Separated
```

## 4. Quick verification with included models

When the blind-test and receiver-like image folders are available, run:

```matlab
run("matlab/step07_run_all_evaluations.m")
```

This executes:

```text
step05_evaluate_blind_test.m
step05b_evaluate_transfer_learning_blind_test.m
step06_evaluate_cnn_receiver_like.m
step06b_evaluate_transfer_learning_receiver_like.m
```

Expected reference accuracies:

| Model | Blind test | Receiver-like validation |
| --- | ---: | ---: |
| Custom CNN | 92.65% | 84.36% |
| ResNet-18 | 93.50% | 86.69% |

Minor numerical differences may occur across MATLAB releases, hardware execution environments, or regenerated training runs.

## 5. Full regeneration

### Generate training data

```matlab
run("matlab/step01_generate_dataset_wifi_bluetooth.m")
```

Expected output:

```text
data/spectrograms_v3_domain_randomized
```

### Train the custom CNN

```matlab
run("matlab/step02_train_cnn_wifi_bluetooth.m")
```

Expected model:

```text
models/cnn_wifi_bluetooth_v3_domain_randomized.mat
```

### Train ResNet-18

```matlab
run("matlab/step02b_train_transfer_learning_wifi_bluetooth.m")
```

Expected model:

```text
models/resnet18_transfer_learning_wifi_bluetooth.mat
```

### Generate the blind test

```matlab
run("matlab/step03_generate_blind_test.m")
```

Expected output:

```text
data/blind_test_v2_final
```

### Generate receiver-like validation data

```matlab
run("matlab/step04_generate_receiver_like_validation.m")
```

Expected outputs:

```text
data/receiver_like_validation/spectrograms
data/receiver_like_validation/metadata_receiver_like_validation.csv
data/receiver_like_validation/generation_config_receiver_like.csv
results/receiver_like_dataset_preview.png
```

The metadata contains repository-relative paths and controlled generation values including SNR, receiver gain, programmed frequency offset, fine oscillator offset, and clipping ratio.

Raw I/Q storage is disabled by default to keep the generated dataset size manageable. Set `saveRawIQ = true` in `step04_generate_receiver_like_validation.m` to store a limited number of examples per class.

## 6. Evaluation outputs

The evaluation scripts write consistent artifact names under `results/`:

```text
metrics_cnn_blind_test.csv
summary_cnn_blind_test.csv
confusion_matrix_cnn_blind_test.png

metrics_transfer_learning_blind_test.csv
summary_transfer_learning_blind_test.csv
confusion_matrix_transfer_learning_blind_test.png

metrics_cnn_receiver_like.csv
summary_cnn_receiver_like.csv
confusion_matrix_cnn_receiver_like.png

metrics_transfer_learning_receiver_like.csv
summary_transfer_learning_receiver_like.csv
confusion_matrix_transfer_learning_receiver_like.png
```

The corresponding `.mat` files preserve predictions, labels, confusion matrices, and summary tables.

## 7. Dataset integrity checks

The receiver-like generator verifies that all six class folders exist and that each contains exactly 600 images. The evaluation scripts also verify that the dataset class set matches the expected six-class definition before classification.

## 8. Interpretation

The independent blind test measures generalization to separately generated samples from a related synthetic distribution. The receiver-like validation set introduces a stronger distribution shift through acquisition-inspired impairments. It therefore provides a more difficult robustness assessment and is not used for model training.
