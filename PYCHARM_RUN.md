# PyCharm Run Guide

Use the interpreter:

`E:\ana\envs\pytorch\python.exe`

Run configurations already added:

- `PolarFree Lite Smoke`
- `PolarFree Lite Train`
- `PolarFree Lite Eval`

Recommended order:

1. Open this project in PyCharm.
2. Set the project interpreter to `E:\ana\envs\pytorch\python.exe` if PyCharm prompts.
3. Run `PolarFree Lite Smoke` once to verify the data matching, cache, GPU training, checkpoint, and test output path.
4. Run `PolarFree Lite Train` for the full quality-first training on `D:\jibi\train`.
5. Run `PolarFree Lite Eval` after training; it evaluates `D:\jibi\data\test` with the latest `best_model.pth`.

Current defaults:

- Train root: `D:\jibi\train`
- Test root: `D:\jibi\data\test`
- Full train: `384`, batch `2`, gradient accumulation `2`, `4` workers, tensor package cache enabled
- Smoke: `128`, batch `2`, `8` train/test samples, `1` epoch

Speed note:

- The first restarted epoch will build `D:\jibi\train\.polarfree_tensor_cache`; later epochs reuse it and should feed the GPU faster.
- If batch `2` causes CUDA out-of-memory, change back to `--batch_size 1 --grad_accum_steps 4`.

If PyTorch still fails to import inside PyCharm, re-select the interpreter as a Conda environment instead of a plain system interpreter path.
