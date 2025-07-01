# Grid Search for CNO and FNO Models

This script performs a comprehensive grid search over hyperparameters for both CNO and FNO architectures on the Helmholtz equation.

## Parameters Swept

- **Learning rates**: 1e-4, 1e-5, 1e-6
- **Boundary weights**: 1, 10, 100, 1000  
- **Lambda values**: 100, 1000, 10000

This results in **3 × 4 × 3 = 36** parameter combinations for each architecture, totaling **72 experiments**.

## Usage

### Running the Grid Search

```bash
python grid_search_helmholtz.py
```

This will:
1. Create a `TrainedModels/helmholtz_gridsearch/` directory
2. Run all 72 experiments sequentially
3. Save each model in a folder named with its parameters, e.g.:
   - `CNO_lr_0.0001_bw_1_lambda_100/`
   - `FNO_lr_1e-05_bw_10_lambda_1000/`

### Analyzing Results

After the grid search completes, analyze the results:

```bash
python grid_search_helmholtz.py analyze
```

This will:
1. Parse all experiment folders
2. Extract the best testing error from each experiment
3. Create a summary CSV file: `TrainedModels/helmholtz_gridsearch/grid_search_summary.csv`
4. Display the top 5 results for both CNO and FNO models

## Output Structure

```
TrainedModels/helmholtz_gridsearch/
├── CNO_lr_0.0001_bw_1_lambda_100/
│   ├── model.pkl
│   ├── errors.txt
│   ├── training_properties.txt
│   ├── net_architecture.txt
│   └── Losses.png
├── FNO_lr_0.0001_bw_1_lambda_100/
│   └── ...
├── ...
└── grid_search_summary.csv
```

## Configuration

The script uses the following base configurations:

### CNO Base Parameters
- 100 epochs
- Batch size: 16
- exp: 1 (L1 loss)
- Training samples: 1024

### FNO Base Parameters  
- 30 epochs
- Batch size: 16
- exp: 3 (smooth loss)
- Training samples: 1024
- Gradient clipping: 5

## Time Estimates

- **CNO experiments**: ~1 hour each (100 epochs)
- **FNO experiments**: ~30 minutes each (30 epochs)
- **Total estimated time**: ~45-50 hours for all experiments

## Error Handling

The script includes:
- Timeout protection (1 hour for CNO, 30 minutes for FNO)
- Error capture and logging
- Graceful handling of failed experiments
- Progress tracking with experiment counters

## Dependencies

- `torch`
- `pandas` (for result analysis)
- `numpy`
- `matplotlib`
- All dependencies from the original training scripts 