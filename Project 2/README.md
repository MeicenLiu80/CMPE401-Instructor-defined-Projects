# CMPE401-Instructor-defined-Project2
# CMPE 401 Project 2

**Name:** Meicen Liu  
**Student ID:** 80436348  
**Date:** 2026/10/5  


## 1. Task 1 — Reproduce the Baseline
### A. Transformer — Time Series Classification
**Dataset:** FordA  
**Task:** Time Series Classification  
**Model:** Transformer  
#### Model Configuration

| Parameter | Value |
|---|---:|
| Model | Transformer |
| Dataset | FordA |
| Input shape | (500, 1) |
| Transformer blocks | 4 |
| Attention head size | 256 |
| Number of heads | 4 |
| Feed-forward dimension | 4 |
| MLP units | 128 |
| Transformer dropout | 0.25 |
| MLP dropout | 0.4 |
| Learning rate | 0.0001 |
| Batch size | 64 |
| Maximum epochs | 150 |
| Early stopping | Patience = 10 |

#### Training Results
| Metric | Result |
|---|---:|
| Actual epochs | **11** |
| Total parameters | **29,258** |
| Test loss | **0.6930** |
| Test accuracy | **51.59%** |

The Transformer baseline was trained on the FordA dataset for time-series classification. The model used four Transformer blocks, with four attention heads and a head size of 256. It used a batch size of 64 and a learning rate of 0.0001. The model trained for 11 epochs before early stopping. The final test accuracy was **51.59%**, with a test loss of **0.6930**.


### B. LSTM — Time Series Forecasting
**Dataset:** Jena Climate  
**Task:** Time Series Forecasting  
**Model:** LSTM  
#### Model Configuration
| Parameter | Baseline |
|---|---:|
| LSTM hidden units | 32 |
| Batch size | 256 |
| Past history | 720 raw time steps |
| Sampling rate | 6 |
| Effective sequence length | 120 |
| Learning rate | 0.001 |
| Maximum epochs | 10 |
| Output | Temperature |
| Loss | MSE |
| Best validation loss| 0.13218 (Epoch 3)| 
The LSTM baseline was trained on the Jena Climate dataset for time-series forecasting. The model used 32 LSTM units, a batch size of 256, and a learning rate of 0.001. The best validation loss was **0.13218**, achieved at epoch 3.


## 2. Task 2 — LSTM Modifications
For Task 2, we selected the LSTM model and made three meaningful modifications.
### Modification 1 — Increase LSTM Hidden Size
**Change:** LSTM hidden size: **32 → 64**
**Best validation loss:** **0.15083 (Epoch 1)**
| | Validation Loss |
|---|---:|
| Baseline | 0.13218 |
| Modification 1 | 0.15083 |
Increasing the LSTM hidden size from 32 to 64 did not improve the validation performance. The best validation loss increased from 0.13218 to 0.15083.

### Modification 2 — Reduce Batch Size
**Change:** Batch size: **256 → 128**
The LSTM hidden size was restored to the baseline value of 32.
**Best validation loss:** **0.12248 (Epoch 10)**
| | Validation Loss |
|---|---:|
| Baseline | 0.13218 |
| Modification 2 | **0.12248** |
Reducing the batch size from 256 to 128 improved the validation performance. The best validation loss decreased from 0.13218 to 0.12248.

### Modification 3 — Increase Input History
**Change:** Past history: **720 → 1440 time steps**
Other parameters were restored to the baseline,LSTM hidden size = 32,Batch size = 256
**Best validation loss:** **0.13500 (Epoch 2)**
| | Validation Loss |
|---|---:|
| Baseline | 0.13218 |
| Modification 3 | 0.13500 |
Increasing the input history from 720 to 1440 time steps did not improve the validation performance. The best validation loss increased slightly from 0.13218 to 0.13500.

### Task 2 Summary
| Model | Modification | Best Validation Loss | Best Epoch | Compared with Baseline |
|---|---|---:|---:|---|
| LSTM Baseline | None | 0.13218 | 3 | — |
| Modification 1 | Hidden size: 32 → 64 | 0.15083 | 1 | Worse |
| Modification 2 | Batch size: 256 → 128 | **0.12248** | 10 | **Best** |
| Modification 3 | History: 720 → 1440 | 0.13500 | 2 | Slightly worse |
In Task 2, we selected the LSTM model and changed three parameters: the LSTM hidden size, the batch size, and the input history. Among the three modifications, reducing the batch size from 256 to 128 gave the best result. The validation loss improved from **0.13218 to 0.12248**.


## 3. Task 3 & Task 4 — Benchmark and Reflection
### Benchmark Results
| Model | Modification | Best Validation Loss | Observation |
|---|---|---:|---|
| LSTM Baseline | None | 0.13218 | Baseline |
| Modification 1 | Hidden size: 32 → 64 | 0.15083 | Performance decreased |
| Modification 2 | Batch size: 256 → 128 | **0.12248** | **Best performance** |
| Modification 3 | Input history: 720 → 1440 | 0.13500 | Slightly decreased |

### Which model did you find easier to understand and why?
For me, the LSTM model was easier to understand because its structure is simpler. The main idea is to input a sequence into the LSTM and use its output to predict a future value. In comparison, the Transformer contains additional components such as multi-head attention, residual connections, normalization, and multiple Transformer blocks, which made it more difficult for me to understand.

### What improvement did you try, and what did you learn from it?
I tried three changes to the LSTM model: increasing the hidden size, reducing the batch size, and increasing the input history. Among the three changes, reducing the batch size from 256 to 128 gave the best result. It improved the validation loss from **0.13218 to 0.12248**.
Overall, a bigger or more complicated model does not always mean better performance. This experiment also showed that training parameters, such as batch size, can have a significant effect on the results.


## Conclusion
This project helped me understand the difference between time-series classification and forecasting, as well as the differences between Transformer and LSTM models. 
After reproducing the two baseline models, I performed three controlled modifications to the LSTM model. 
The best result was obtained by reducing the batch size from 256 to 128, which improved the validation loss from 0.13218 to 0.12248.
