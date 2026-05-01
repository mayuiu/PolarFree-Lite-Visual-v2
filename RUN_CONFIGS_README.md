# Run Configs

## PyCharm

Open `C:\Users\86166\Desktop\PythonProject` in PyCharm.

To run manually from PyCharm:

1. Open `polarfree_lite_glass_v2.py`.
2. Choose the Python interpreter `E:\ana\envs\pytorch\python.exe`.
3. Set the working directory to `C:\Users\86166\Desktop\PythonProject`.
4. Add the parameters you want in the Run Configuration.
5. Click the green Run button.

Existing `.idea/runConfigurations` entries may also appear in the top-right selector. If PyCharm does not pick them up, use the `.bat` files below.

## Bat Files

Double-click one of these files from the project root:

- `run_test_full_raw.bat`
- `run_test_full_visual.bat`
- `run_train_fast_effect.bat`
- `run_train_fast_tiny.bat`
- `run_train_fast_gpu_boost.bat`
- `run_train_fast_50ep.bat`
- `run_train_full_80ep.bat`
- `run_analyze_full_raw_latest.bat`
- `run_analyze_full_visual_latest.bat`

Each script changes to the project directory, activates the `pytorch` conda environment, runs the command, then pauses.

## Presets

`fast_effect` is for quick visible feedback:

- `img_size=384`
- `batch_size=1`
- `grad_accum_steps=4`
- `epochs=15`
- `lr=8e-5`
- `eval_every=3`
- `test_limit=20`
- `num_workers=2` unless manually overridden
- `render_mode=raw`
- `mask_profile=balanced`
- `loss_profile=balanced`

`full_stable` is more conservative:

- `img_size=384`
- `batch_size=1`
- `grad_accum_steps=4`
- `epochs=30`
- `lr=6e-5`
- `eval_every=5`
- `test_limit=30`
- `num_workers=2` unless manually overridden
- `render_mode=raw`
- `mask_profile=stable`
- `loss_profile=stable`

`fast_reflection` is the recommended quick reflection check:

- `img_size=384`
- `batch_size=1`
- `grad_accum_steps=4`
- `epochs=10`
- `lr=8e-5`
- `eval_every=3`
- `test_limit=20`
- `num_workers=0`
- `render_mode=raw`
- `mask_profile=balanced`
- `loss_profile=balanced`

`gpu_boost` is optional for higher GPU utilization:

- `batch_size=2`
- `grad_accum_steps=2`
- `num_workers=2`
- `prefetch_factor=2`
- `eval_every=5`
- `test_limit=10`
- `render_mode=raw`

If `gpu_boost` runs out of memory or hangs on Windows, go back to `batch_size=1` and `num_workers=0`.

If you pass a value manually, the preset will not override it. For example, `--num_workers 0` stays `0`.

## Raw vs Visual

- `raw` is the model Raw output and is currently the recommended default.
- `visual` applies visual post-processing. In the full test result, visual was less stable than raw.

## Output Directories

- Full raw test: `output\glass_v2_full_test_raw`
- Full visual test: `output\glass_v2_full_test_visual`
- Fast hard-case training: `output\fast_effect_hardcase`
- Fast tiny training: `output\fast_effect_tiny`
- GPU boost reflection training: `output\fast_reflection_gpu_boost`
- 50 epoch fast training: `output\glass_v2_fast_50ep`
- 80 epoch full training: `output\glass_v2_80ep_full_power`

Evaluation now also writes:

- `per_sample_metrics.csv`
- `per_sample_metrics.json`
- `worst_cases.csv`

These are saved inside the active `test_results` directory.

## Do Not Delete

Do not delete:

- `main.py`
- `polarfree_lite.py`
- `polarfree_lite_glass_v2.py`
- `AGENTS.md`
