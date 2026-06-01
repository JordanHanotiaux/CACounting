# CACounting

## Dataset — FSC-147

Download from [LearningToCountEverything](https://github.com/cvlab-stonybrook/LearningToCountEverything) and place as:

```
raid/datasets/FSC147_384_V2/
├── images_384_VarV2/
├── gt_density_map_adaptive_384_VarV2/
├── annotation_FSC147_384.json
└── Train_Test_Val_FSC_147.json
```

## Usage

```bash
# Evaluate on FSC-147 test split
python evaluation.py --split test
```

### Key arguments — `evaluation.py`

| Argument | Default | Description |
|---|---|---|
| `--split` | `test` | `train`, `val`, `test` |
| `--percentage` | `100` | % of split to evaluate |
| `--output_dir` | `eval_results` | Output directory |
| `--top_best_n` | `20` | Best predictions to visualise |
| `--top_worst_n` | `10` | Worst predictions to visualise |

Results are saved to `output_dir/predictions.csv` with per-image MAE, RMSE and diagnostic flags.
