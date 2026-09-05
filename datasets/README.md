# Dataset Preparation Guide

## Why you need this

`models/yolo11n.pt` (and `yolov8n.pt`) are **COCO-pretrained** checkpoints.
COCO only contains 10 generic animal categories: bird, cat, dog, horse, sheep,
cow, elephant, bear, zebra, giraffe. There is no "tiger", "leopard", "deer",
"wild boar", "wolf", etc. To detect real wildlife species you must fine-tune
YOLO on your own labeled images of those species. This is normal — every
production wildlife-CV system does this; there is no shortcut around needing
labeled data.

## 1. Collect images

Gather images of each species you want to detect, from as many angles,
lighting conditions (day/night/IR), distances, and backgrounds as you can.
Guidelines:
- **Minimum viable**: ~150-300 images per species to get a working prototype.
- **Solid production model**: 1,000-2,000+ images per species.
- Include hard cases: partial occlusion, animals at the frame edge, multiple
  animals overlapping, motion blur, night/IR footage if your cameras have it.

Good starting sources for pretraining/augmenting your own camera footage:
- Your own camera trap / CCTV footage (best — matches your real deployment conditions)
- **LILA BC** (lila.science) — large collection of labeled camera-trap datasets
- **iNaturalist** (inaturalist.org) — huge species-labeled photo database
- **Roboflow Universe** (universe.roboflow.com) — many pre-labeled wildlife datasets
  in YOLO format already, searchable by species
- **Open Images / Kaggle** — general animal datasets, useful for pretraining

Always check each dataset's license before using it in anything commercial.

## 2. Label the images (YOLO format)

Each image needs a matching `.txt` label file with the **same base filename**:

```
images/lion_001.jpg
labels/lion_001.txt
```

Each line in the `.txt` file is one object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: integer starting at 0, matching the order in `data.yaml` `names`
- All 4 coordinates are **normalized 0-1** (fraction of image width/height),
  NOT pixel values
- One line per animal in the image; empty `.txt` file = no animals present

Example (`lion_001.txt`, image is 1920x1080, one lion roughly centered):
```
0 0.512 0.487 0.310 0.420
```

### Labeling tools
- **LabelImg** (github.com/HumanSignal/labelImg) — simple, free, exports YOLO format directly
- **CVAT** (cvat.ai) — free, browser-based, good for video frame-by-frame labeling
- **Roboflow** (roboflow.com) — free tier, browser-based, auto-augmentation, exports YOLO format
- **makesense.ai** — free, no signup, browser-based

Any of these can export directly in the YOLO `.txt` format described above.

## 3. Organize your raw data

Before running `prepare_dataset.py`, put everything in one flat structure:

```
datasets/raw/
├── images/
│   ├── lion_001.jpg
│   ├── lion_002.jpg
│   ├── leopard_001.jpg
│   └── ...
└── labels/
    ├── lion_001.txt
    ├── lion_002.txt
    ├── leopard_001.txt
    └── ...
```

Also write a `classes.txt` listing your species, one per line, in the exact
order your `class_id`s refer to:

```
datasets/raw/classes.txt
```
```
lion
leopard
deer
wild_boar
wolf
```

## 4. Run the preparation script

```bash
python -m datasets.prepare_dataset \
    --images datasets/raw/images \
    --labels datasets/raw/labels \
    --classes datasets/raw/classes.txt \
    --output datasets/wildlife_dataset
```

This will:
- Validate every image has a matching label file (and warn about any that don't)
- Split into train/val/test (80/10/10 by default, configurable in `configs/config.yaml` -> `training.train_val_test_split`)
- Copy files into `datasets/wildlife_dataset/{images,labels}/{train,val,test}/`
- Generate `datasets/wildlife_dataset/data.yaml`, which is what the trainer reads

## 5. Train

See `training/train_yolo.py` — in short:

```bash
python -m training.train_yolo --data datasets/wildlife_dataset/data.yaml --epochs 100
```

## 6. Evaluate

See `training/evaluate.py` — produces precision/recall/mAP metrics and a
confusion matrix:

```bash
python -m training.evaluate --weights outputs/training_runs/wildlife_yolo/weights/best.pt \
    --data datasets/wildlife_dataset/data.yaml
```

## 7. Use your trained model

Point `configs/config.yaml` -> `detection.model_path` at your new weights:

```yaml
detection:
  model_path: "outputs/training_runs/wildlife_yolo/weights/best.pt"
```

`training/train_yolo.py` automatically updates `configs/species_map.yaml` ->
`custom_wildlife` with your trained class names, so species labels show up
correctly in the dashboard, API, and database right away.
