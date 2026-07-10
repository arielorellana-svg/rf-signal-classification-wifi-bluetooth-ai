# Pre-generated Receiver-Like Validation Set

This directory contains the pre-generated synthetic receiver-like spectrogram set used for the reported robustness results.

Dataset characteristics:

- Six RF classes
- 600 images per class
- 3,600 spectrograms in total
- Synthetic complex I/Q generation
- Receiver-inspired impairments and controlled metadata

The canonical regeneration script is:

```matlab
run("matlab/step04_generate_receiver_like_validation.m")
```

Newly generated data is written to:

```text
data/receiver_like_validation/spectrograms
```

The evaluation scripts detect both this pre-generated copy and the canonical generated location.
