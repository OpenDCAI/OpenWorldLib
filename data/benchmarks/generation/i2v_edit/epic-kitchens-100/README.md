# EPIC-KITCHENS → i2v Dataset Pipeline

We provide a small set of test cases hosted on HuggingFace Datasets.
you can run:

```bash
git clone https://huggingface.co/datasets/YF0224/demo
```

## Step 1. Download EPIC-KITCHENS-100 Videos

Clone and use the official download scripts:

```bash
git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
```

Download RGB videos and organize them as:

```
data/epic-kitchens/
├── videos_raw/
│   ├── P01_01.mp4
│   ├── P01_02.mp4
│   └── ...
```

---

## Step 2. Download EPIC-KITCHENS-100 Annotations

Clone the annotation repository:

```bash
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git
```

Use the narration/action CSV files and place them as:

```
data/epic-kitchens/
├── annotations.csv
```

---

## Step 3. Clip Videos to Build i2v Samples

Run the preprocessing script to clip videos according to annotations:

```bash
python build_epic_i2v.py
```

The script will:

* read `annotations.csv`
* clip video segments from `videos_raw`
* save the first frame as `init_frame.jpg`
* generate per-sample folders and a unified `samples.json`

---

## Output Structure

```
data/epic-kitchens/
├── samples/
│   ├── P01_01_0/
│   │   ├── init_frame.jpg
│   │   └── P01_01_0.mp4
│   ├── ...
├── samples.json
```

---
