**Electricity Load Forecasting with Transformer & LSTM**

This repository implements end-to-end time-series forecasting of household electricity consumption using both Transformer and LSTM models. The pipeline covers data preprocessing, feature engineering, sequence construction, model training, evaluation, and visualization.

## Repository Structure

```
├── transformer.py            # Transformer model implementation and training
├── lstm.py                   # LSTM model implementation and training
├── household_power_consumption.txt  # Raw power consumption data (semicolon-separated)
├── pictures.png              # Generated plots (load curves, correlation heatmaps, loss curves, predictions)
└── README.md                 # This documentation
```

## Dependencies

The code relies on the following Python packages:

- pandas
- numpy
- matplotlib  (uses TkAgg backend by default)
- seaborn
- scikit-learn
- torch (PyTorch)
- tqdm

Install via:

```bash
pip install -r requirements.txt
```

## Data Preparation

Both scripts (`transformer.py` and `lstm.py`) execute these preprocessing steps:

1. **Loading & Cleaning**
   - Read `household_power_consumption.txt` (semicolon-separated, `?` or `NaN` as missing).
   - Drop rows with missing values.
   - Merge `Date` + `Time` into a `Datetime` index.
2. **Outlier Removal**
   - Apply a 3&sigma; rule on `Global_active_power`.
3. **Feature Engineering**
   - Extract `hour` and `weekday` from the datetime index.
   - Compute a 60-minute rolling mean of `Global_active_power` (`global_active_ma60`).
4. **Sequence Construction**
   - Use a sliding window of length 60 to build input sequences (features) and one-step-ahead targets.
5. **Normalization**
   - Fit a separate `StandardScaler` for each feature and the target.
6. **Train/Val/Test Split**
   - Split sequences chronologically into 50% train, 25% validation, and 25% test sets.

After preprocessing, each script proceeds with its respective model.

## Transformer Model (`transformer.py`)

1. **Architecture**
   - Feature embedding → positional encoding → 3-layer TransformerEncoder (d_model=64, 4 heads, dropout=0.1).
   - A final linear decoder projects to one output value.
2. **Training Setup**
   - Loss: MSELoss
   - Optimizer: Adam (lr=1e-3)
   - Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=2)
   - Early stopping on validation loss (patience=5)
   - Epochs: up to 50
   - Batch size: 64
3. **Output**
   - Plots saved in working directory:
     - `load_curve.png` (first 1000 points)
     - `correlation_heatmap.png`
     - `transformer_loss_curve1.png`
     - `transformer_predictions1.png`
   - Prints final Test MSE, MAE, RMSE.

**Run:**
```bash
python transformer.py
```

## LSTM Model (`lstm.py`)

1. **Architecture**
   - 2-layer LSTM (hidden_size=64, dropout=0.1) with batch-first input
   - Fully connected layer to one output
2. **Training Setup**
   - Loss: MSELoss
   - Optimizer: Adam (lr=1e-3)
   - Scheduler: `StepLR` (step_size=10, gamma=0.5)
   - Early stopping on validation loss (patience=5)
   - Epochs: up to 30
   - Batch size: 64
3. **Output**
   - Plots saved in working directory:
     - `load_curve1.png` (first 1000 points)
     - `correlation_heatmap1.png`
     - `lstm_loss_curve1.png`
     - `lstm_predictions1.png`
   - Prints final Test MSE, MAE, RMSE.

**Run:**
```bash
python lstm.py
```

## Configuration & Reproducibility

- Both scripts set a fixed random seed (`42`) for Python, NumPy, and PyTorch, and disable CuDNN nondeterminism to ensure reproducibility.
- The Matplotlib backend is set to `TkAgg` to avoid environment issues; adjust as needed.

## Parameters

You can modify key hyperparameters at the top of each script:

- `window_size`: input sequence length (default 60)
- Transformer settings: `d_model`, `nhead`, `num_layers`, `dropout`
- LSTM settings: `hidden_size`, `num_layers`, `dropout`
- Training: `batch_size`, `num_epochs`, learning rates, scheduler options

Adjust and re-run to experiment with different configurations.

## Notes

- Ensure `household_power_consumption.txt` is in the same directory as the scripts.
- GPU acceleration is used automatically if available.
- Training may take several minutes depending on hardware.