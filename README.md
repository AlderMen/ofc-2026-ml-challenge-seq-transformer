# OFC 2026 ML Challenge: Sequence Transformer Solution

This repository contains the Sequence Transformer solution for the **OFC 2026 Machine Learning Challenge**. The model is designed to predict EDFA gain profiles with high precision, utilizing multi-scale tokens and local unloaded context.

## Overview

The solution employs a Transformer-based architecture to process channel spectra and EDFA parameters. Key features include:
- **Sequence Modeling**: Treating the 95-channel spectrum as a sequence.
- **Ensemble Learning**: Combining multiple models with different seeds for robust predictions.
- **Two-Stage Training**: Utilizing both broad datasets (COSMOS) and competition-specific data.
- **Residual Prediction**: Predicting gains relative to target priors for improved stability.

## Repository Structure

- `main.py`: The core script for training and inference.
- `requirements.txt`: List of required Python packages.
- `run_pipeline.ps1`: PowerShell script to execute the full training and prediction pipeline.
- `.gitignore`: Standard patterns to exclude from version control.

## Installation

Ensure you have Python 3.8+ installed. Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Data Preparation

Place the competition data in the following structure (or update the paths in `main.py`):

```text
ofc-ml-challenge-data-code-main/
  Features/
    Train/
      train_features.csv
      train_labels.csv
    Test/
      test_features.csv
```

## How to Run

To run the full pipeline (Training + Prediction) as configured for our submission:

### PowerShell (Windows)

```powershell
./run_pipeline.ps1
```

### Manual Command

```powershell
python main.py --train --predict --amp --save_best `
  --out_dir ./runs/seqtf_v3/run_residual --tag run_residual `
  --n_ensemble 5 --ensemble `
  --attn_mode pad_unloaded `
  --cosmos_max 10000 --train_mode two_stage `
  --residual_from_target
```

## Methodology

### Model Architecture
The `SeqTransformer` uses a `TransformerEncoder` to capture inter-channel dependencies. It incorporates global and optional pooled segment tokens to aggregate spectral information.

### Loss Function
We support standard Mean Squared Error (MSE) and a custom `kaggle_proxy` loss that optimizes for the competition's evaluation metrics (MAE, Std, and P95 error).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OFC 2026 ML Challenge Organizers
