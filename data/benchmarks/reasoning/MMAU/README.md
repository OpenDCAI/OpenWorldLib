# **MMAU — Benchmark Record (Minimal)**

## **1. Meta**

* **Name**: MMAU (Massive Multi-Task Audio Understanding)
* **Task**: Multi-task audio understanding & reasoning evaluation
* **Paper**: [https://arxiv.org/pdf/2410.19168](https://arxiv.org/pdf/2410.19168)
* **Homepage**: [https://sakshi113.github.io/mmau_homepage/](https://sakshi113.github.io/mmau_homepage/)
* **Code**: [https://github.com/Sakshi113/MMAU](https://github.com/Sakshi113/MMAU)
* **Benchmark Code Path**: `MMAU/evaluation.py`
* **Task Type**: Audio Understanding / Audio Reasoning / Audio QA

---

## **2. Dataset Format**

### **Directory Structure**

```
$DATA_PATH/MMAU/
├── test-mini/
│   ├── audios/
│   │   ├── mmau_000001.wav
│   │   ├── mmau_000002.wav
│   │   └── ...
│   └── test-mini.json
└── test/
    ├── audios/
    │   ├── mmau_010001.wav
    │   ├── mmau_010002.wav
    │   └── ...
    └── test.json
```

---

### **Sample JSON Entry (Original)**

```json
{
  "id": "mmau_000123",
  "audio_path": "audios/mmau_000123.wav",
  "question": "What instrument is playing after the drum beat?",
  "choices": [
    "Piano",
    "Violin",
    "Guitar",
    "Flute"
  ],
  "answer": "Guitar"
}
```

---

## **3. IO Specification**

### **Input (per instance)**

```python
{
  "audio_path": str,
  "question": str,
  "choices": List[str] | None
}
```

---

### **Output (per instance)**

```python
{
  "model_prediction": str
}
```

---

## **4. Evaluation**

### **Evaluation Script Path**

```
MMAU/evaluation.py
```

---

### **Local Evaluation (test-mini)**

```bash
python evaluation.py \
  --input /path/to/test-mini.json
```

---