# OFC 2026 ML Challenge: Sequence Transformer Solution

**Team:** IPOC  
**Task:** EDFA Gain Profile Prediction (95-channel spectral gain)

## Repository Structure

```
.
├── main.py                          # Training & inference script
├── requirements.txt                 # Python dependencies
├── best_model/
│   ├── checkpoints/
│   │   ├── best_model_0.pth         # Ensemble member 0
│   │   ├── best_model_1.pth         # Ensemble member 1
│   │   ├── best_model_2.pth         # Ensemble member 2
│   │   ├── best_model_3.pth         # Ensemble member 3
│   │   └── best_model_4.pth         # Ensemble member 4
│   └── scalers/
│       └── best_model.npz           # Standardizer + model config
└── README.md
```

## Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Inference (Reproduce Submission)

### Quick Start

The simplest way to run inference on new test data:

```bash
python main.py --predict \
    --out_dir ./best_model \
    --tag best_model \
    --ensemble --n_ensemble 5 \
    --test_csv /path/to/test_features.csv \
    --output_csv ./submission.csv
```

**Arguments:**

| Argument | Description |
|---|---|
| `--predict` | Run inference only (no training) |
| `--out_dir ./best_model` | Directory containing model weights and scalers |
| `--tag best_model` | Model tag (matches checkpoint/scaler filenames) |
| `--ensemble --n_ensemble 5` | Use 5-member ensemble averaging |
| `--test_csv <path>` | Path to the new `test_features.csv` |
| `--output_csv <path>` | Where to write the submission CSV |

### Alternative: Default Data Path

If you place the test data at the default location, `--test_csv` can be omitted:

```
ofc-ml-challenge-data-code-main/
  Features/
    Test/
      test_features.csv
```

```bash
python main.py --predict --out_dir ./best_model --tag best_model --ensemble --n_ensemble 5
```

Output will be saved to `best_model/submissions/submission_best_model.csv`.

### Expected Input Format

The input `test_features.csv` must be a Kaggle-format CSV with columns:

- `ID` (integer)
- `EDFA_input_spectra_00` ... `EDFA_input_spectra_94` (95 input channels)
- `DUT_WSS_activated_channel_index_00` ... `DUT_WSS_activated_channel_index_94` (95 mask channels)
- `target_gain`, `target_gain_tilt`, `EDFA_input_power_total`, `EDFA_output_power_total`
- `EDFA_type`, `edfa_index`, `Category` (optional)

### Expected Output Format

The output CSV follows Kaggle submission format:

- `ID` (integer)
- `calculated_gain_spectra_00` ... `calculated_gain_spectra_94` (95 predicted gain values in dB)

## Model Details

- **Architecture:** Transformer Encoder (d_model=256, depth=8, nhead=8, dim_ff=768)
- **Ensemble:** 5 models with different random seeds, predictions averaged
- **Training strategy:** Two-stage — pretrain on full dataset (Kaggle + COSMOS), finetune on Kaggle-only data
- **Residual prediction:** Model predicts residual relative to target gain prior
- **Attention masking:** Unloaded channels masked (`pad_unloaded` mode)
- All model configuration is automatically loaded from `best_model/scalers/best_model.npz` at inference time; no manual parameter specification is needed.

## License

MIT License — see LICENSE file for details.
