# Dataset Layout

The project uses three synthetic datasets with a shared six-class folder structure.

## Training dataset

```text
data/spectrograms_v3_domain_randomized
```

- 3,000 spectrograms per class
- 18,000 images total
- Random seed: 21
- Used for training, validation, and internal testing

Generate with:

```matlab
run("matlab/step01_generate_dataset_wifi_bluetooth.m")
```

## Independent blind test

```text
data/blind_test_v2_final
```

- 1,000 spectrograms per class
- 6,000 images total
- Random seed: 2026
- Generated independently from the training dataset

Generate with:

```matlab
run("matlab/step03_generate_blind_test.m")
```

## Receiver-like validation

```text
data/receiver_like_validation/spectrograms
```

- 600 spectrograms per class
- 3,600 images total
- Random seed: 407
- Generated independently from training
- Includes acquisition-inspired receiver effects

Generate with:

```matlab
run("matlab/step04_generate_receiver_like_validation.m")
```

Additional files:

```text
data/receiver_like_validation/metadata_receiver_like_validation.csv
data/receiver_like_validation/generation_config_receiver_like.csv
```

The metadata contains controlled synthetic parameters such as SNR, receiver gain, programmed frequency displacement, oscillator mismatch, and clipping ratio. File references are stored relative to the repository root.

## Class folders

```text
Bluetooth
Noise
Unknown
WiFi
WiFi_Bluetooth_Overlap
WiFi_Bluetooth_Separated
```

Large generated datasets are excluded from normal version control. The scripts, pretrained models, metrics, and generation configuration provide the reproducible project definition.
