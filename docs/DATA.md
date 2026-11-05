# Data preparation

SkyPart does not redistribute datasets. Obtain each dataset from its original
provider and follow its access terms. All examples below are relative to the
SkyPart repository root.

## Path overrides

The shipped YAML files use:

```yaml
SUES_ROOT: "./data/sues200"
DATA_ROOT: "./data/university1652"
DENSEUAV_ROOT: "./data/denseuav"
OUTPUT_DIR: "./outputs"
```

Paths are resolved from the command's working directory. To store data
elsewhere without changing a tracked recipe:

```bash
mkdir -p configs/local
cp configs/sues200/default.yaml configs/local/sues200.yaml
```

Edit only the copied root/output fields, using paths relative to this project
root, then run `python scripts/train.py --config configs/local/sues200.yaml`.
Leave the model and protocol fields unchanged.

## SUES-200

The loader expects 120 training and 80 test locations, four altitudes, and a
200-tile satellite confusion gallery:

```text
data/sues200/
  drone-view/
    <four-digit-location>/
      150/<image>.png
      200/<image>.png
      250/<image>.png
      300/<image>.png
  satellite-view/
    <four-digit-location>/0.png
```

Training uses locations 1--120; testing queries locations 121--200. Clean and
weather evaluation use all 200 satellite tiles as the D-to-S gallery.

## University-1652

The loader implements the official asymmetric four-folder test protocol:

```text
data/university1652/
  train/
    drone/<building-id>/<image>.jpg
    satellite/<building-id>/<image>.jpg
  test/
    query_drone/<building-id>/<image>.jpg
    gallery_satellite/<building-id>/<image>.jpg
    query_satellite/<building-id>/<image>.jpg
    gallery_drone/<building-id>/<image>.jpg
```

There is no altitude metadata. The YAML uses `NUM_ALTITUDES: 1`, and inference
uses the mean FiLM behavior. Folder names must be integer building IDs for
evaluation label parsing.

## DenseUAV

Training pairs three heights per location and ignores filenames containing
`_old`. Evaluation follows the official protocol: every image in
`gallery_satellite` (all locations, `_old` variants included) is the D-to-S
gallery, and `query_satellite` / `gallery_drone` give the S-to-D direction:

```text
data/denseuav/
  train/
    drone/<location>/H80.JPG
    drone/<location>/H90.JPG
    drone/<location>/H100.JPG
    satellite/<location>/H80.tif
    satellite/<location>/H90.tif
    satellite/<location>/H100.tif
  test/
    query_drone/<location>/H80.JPG
    query_drone/<location>/H90.JPG
    query_drone/<location>/H100.JPG
    gallery_satellite/<location>/H80.tif
    gallery_satellite/<location>/H90.tif
    gallery_satellite/<location>/H100.tif
    gallery_satellite/<location>/H80_old.tif   (and H90_old, H100_old)
    query_satellite/<location>/...
    gallery_drone/<location>/...
  Dense_GPS_ALL.txt
```

Extension matching is case-insensitive for supported image formats. The GPS
file is optional for retrieval: when absent, SDM@1/3/5 values are serialized as
null while R@1/5/10 and AP remain available. `DENSE_GPS_FILE` is a filename
resolved under `DENSEUAV_ROOT`.

## WeatherPrompt inputs

Weather evaluation needs no second dataset. The ten conditions are generated
online for drone images while satellite images remain clean. The implementation
is in [`skypart/data/weather.py`](../skypart/data/weather.py) and requires
`imgaug`.

Before a long run, confirm that each expected split is discovered:

```bash
python scripts/train.py --config configs/sues200/default.yaml --seed 42
```

The loader prints sample and class counts before model training. Stop the run
if they do not match the intended dataset distribution; the code does not
download, checksum, or license-check dataset content.
