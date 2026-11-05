# Training and evaluation

Run every command from the repository root. Data layout is described in
[DATA.md](DATA.md).

## Setup

```bash
conda env create -f environment.yml
conda activate skypart
pip install -e .
python scripts/smoke_test.py   # synthetic forward/backward, no dataset needed
```

The DINOv2 student (ViT-S/14) and teacher (ViT-B/14) are fetched from Torch Hub
on first use, pinned to `facebookresearch/dinov2@7764ea0`.

## Training

Clean recipes (448×448, 60 epochs):

```bash
python scripts/train.py --config configs/sues200/default.yaml
python scripts/train.py --config configs/university1652/default.yaml
python scripts/train.py --config configs/denseuav/default.yaml
```

Weather recipes fine-tune the clean checkpoint for 60 epochs at 392×392 with
online WeatherPrompt corruptions on the drone view only:

```bash
python scripts/train.py --config configs/sues200/weather.yaml --weather \
    --init-ckpt outputs/skypart_sues200_final.pth
python scripts/train.py --config configs/university1652/weather.yaml --weather \
    --init-ckpt outputs/skypart_university1652_final.pth
python scripts/train.py --config configs/denseuav/weather.yaml --weather \
    --init-ckpt outputs/skypart_denseuav_final.pth
```

Rotation augmentation ({0, 90, 180, 270} on both views) is applied to DenseUAV
only, whose satellite crops have arbitrary basemap orientation. The
altitude-supervision term is dropped on University-1652, which has no altitude
labels.

The paper reports the mean over seeds 42, 123 and 2024. Use `--seed` together
with a separate output directory per seed, since file names are shared within
a recipe:

```bash
for s in 42 123 2024; do
  python scripts/train.py --config configs/sues200/default.yaml --seed $s
  mkdir -p outputs/seed_$s && mv outputs/skypart_sues200_* outputs/seed_$s/
done
```

Training evaluates on the test split every `EVAL_INTERVAL` epochs for logging
only. The saved model is always the final-epoch checkpoint; no checkpoint is
selected on test data. Each run writes

```text
outputs/skypart_<dataset>[_weather]_final.pth
outputs/skypart_<dataset>[_weather]_summary.json
```

where the summary holds the full config, model complexity and the evaluation
history.

## Evaluation

All evaluation is a single forward pass with cosine retrieval, no re-ranking
and no test-time augmentation. No altitude metadata is used at inference: both
views are embedded with the mean FiLM parameters.

```bash
python scripts/eval.py --config configs/sues200/default.yaml --ckpt <checkpoint.pth>
python scripts/eval_weather.py --config configs/sues200/weather.yaml --ckpt <checkpoint.pth>
```

Replace `sues200` with `university1652` or `denseuav` for the other benchmarks,
and use the config that matches the checkpoint's training recipe.

- **SUES-200**: D→S per altitude and overall against the full 200-tile
  gallery, plus S→D; R@1/5/10 and AP.
- **University-1652**: the official four-folder D→S and S→D protocol;
  R@1/5/10 and AP.
- **DenseUAV**: the official protocol. D→S uses the full `gallery_satellite`
  (all locations and heights), S→D uses `query_satellite` against
  `gallery_drone`; R@1/5/10, AP, and SDM@1/3/5 when `Dense_GPS_ALL.txt` is
  present.
- **Weather**: ten WeatherPrompt conditions applied to the drone images in both
  directions, satellite kept clean; per-condition results and their mean.

Results are written to `OUTPUT_DIR` as JSON.

## Checkpoint format

Checkpoints store the model weights together with the training config and a
fingerprint of the architecture fields. `eval.py` loads them with
`torch.load(..., weights_only=True)`, rejects a checkpoint whose architecture
or DINOv2 identity does not match the chosen config, and then loads the
weights strictly. A plain state dict without this metadata is refused unless
`--allow-legacy-checkpoint` is passed.
