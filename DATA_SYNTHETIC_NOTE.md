# Synthetic Data Note

This repository is prepared for a course project, not for real-world forecasting.

- All input tables in `data/` should be treated as synthetic, illustrative, or manually assembled demo inputs.
- `data/synthetic_target_influenza_2025_2026.csv` is a synthetic weekly target series used only to exercise the calibration workflow.
- `Region_A` and `Region_B` are placeholder labels for a two-region model.
- Legacy names containing `observed` or `actualfit` remain in a few files for backward compatibility. They do not imply official or validated real-world data.

Recommended submission-time commands:

```powershell
python -m src.cli --config configs/calendar_baseline.yaml --mode calendar --run-name winter_calendar_demo
python -m src.calibrate_cli --config configs/calendar_baseline.yaml --target-csv data/synthetic_target_influenza_2025_2026.csv --run-name synthetic_target_calibration
python -m src.cli --config configs/calendar_synthetic_calibrated.yaml --mode calendar --run-name winter_calendar_synthetic_calibrated
python -m src.cli --config configs/calendar_fullseason_extended_demo_fit.yaml --mode calendar --run-name fullseason_extended_demo_fit
```

Recommended presentation wording:

- "This model was tested with synthetic population/contact inputs and a synthetic weekly target series."
- "The calibration step is a shape-matching demo against synthetic targets, not a fit to official surveillance records."
- "The outputs are intended to demonstrate model behavior and scenario comparison only."
