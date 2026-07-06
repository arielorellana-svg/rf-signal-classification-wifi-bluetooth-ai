\# Validation Summary



This project classifies RF signals using MATLAB-generated I/Q waveforms, spectrograms, and a convolutional neural network.



\## Classes



\- Bluetooth

\- Noise

\- Unknown

\- WiFi

\- WiFi\_Bluetooth\_Overlap

\- WiFi\_Bluetooth\_Separated



\## Dataset Summary



| Dataset | Classes | Images per class | Total images |

|---|---:|---:|---:|

| Domain-randomized training dataset | 6 | 3,000 | 18,000 |

| Final independent blind-test dataset | 6 | 1,000 | 6,000 |



\## Validation Results



| Evaluation stage | Accuracy |

|---|---:|

| Internal V3 test | 95.52% |

| Blind test V1 | 94.50% |

| Final independent blind test | 92.65% |



\## Final Independent Test Metrics



| Class | Precision | Recall | F1-score |

|---|---:|---:|---:|

| Bluetooth | 0.9115 | 0.9270 | 0.9192 |

| Noise | 0.9861 | 0.9950 | 0.9905 |

| Unknown | 0.9969 | 0.9520 | 0.9739 |

| WiFi | 0.8104 | 0.9790 | 0.8868 |

| WiFi\_Bluetooth\_Overlap | 0.9437 | 0.7880 | 0.8589 |

| WiFi\_Bluetooth\_Separated | 0.9406 | 0.9180 | 0.9292 |



\## Confusion Matrix



The final independent confusion matrix is available in:



```text

results/confusion\_matrix\_blind\_test\_v3\_final.png

