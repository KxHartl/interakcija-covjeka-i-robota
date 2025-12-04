# Brza uputa - Zadatak 1

## Brzo pokretanje (5 koraka)

### 1. Kloniraj repozitorij

```bash
git clone https://github.com/KxHartl/interakcija-covjeka-i-robota.git
cd interakcija-covjeka-i-robota/skeletonizacija
```

### 2. Instaliraj zavisnosti

```bash
pip install -r requirements.txt
```

### 3. Preuzmi PennAction dataset

**Opcija A - Ručno preuzimanje:**
1. Idi na: http://dreamdragon.github.io/PennAction/
2. Preuzmi `Penn_Action.tar.gz`
3. Ekstrahiraj u `data/raw/PennAction/`

**Opcija B - Wget (Linux/Mac):**
```bash
mkdir -p data/raw/PennAction
cd data/raw/PennAction
wget http://dreamdragon.github.io/PennAction/Penn_Action.tar.gz
tar -xzf Penn_Action.tar.gz
cd ../../..
```

### 4. Prilagodi odabir slika (opcionalno)

Uredi `src/zadatak1_compare_gt_mediapipe.py`, linija ~200:

```python
selected_images = [
    ("0001", 50),   # (video_id, frame_idx)
    ("0100", 30),
    # ... dodaj svoje odabrane slike
]
```

### 5. Pokreni

```bash
cd src
python zadatak1_compare_gt_mediapipe.py
```

---

## Output

Rezultati se spremaju u `data/results/zadatak1/`:

- `summary_results.csv` - Tablica s metrikama
- `comparison_XXXX_frameYY.jpg` - Vizualizacije

---

## Prikaz rezultata

### Konzolni output primjer:

```
============================================================
ZADATAK 1: Usporedba Ground Truth i MediaPipe Keypoints
============================================================

Procesuiranje: ../data/raw/PennAction/frames/0001/000050.jpg
  Ground truth učitan: (13, 2)
  MediaPipe detektirao: (33, 2)
  RMSE: 15.32 px
  Mean error: 14.87 px
  ✓ Vizualizacija spremljena

============================================================
RESULTATI USPOREDBE
============================================================

Slika          | RMSE (px) | Mean Error (px) | ...
000050.jpg     |    15.32  |     14.87       | ...
...

UKUPNA STATISTIKA:
Prosječni RMSE:       16.78 px
Prosječna Mean Error: 15.92 px
============================================================
```

---

## Rješavanje češćih problema

### ModuleNotFoundError: No module named 'cv2'

```bash
pip install opencv-python
```

### FileNotFoundError: .mat datoteka ne postoji

Provjeri strukturu PennAction dataseta:

```
data/raw/PennAction/
  ├── frames/
  │   ├── 0001/
  │   │   ├── 000001.jpg
  │   │   ├── 000002.jpg
  │   │   └── ...
  │   ├── 0100/
  │   └── ...
  └── labels/
      ├── 0001.mat
      ├── 0100.mat
      └── ...
```

### MediaPipe ne detektira pozu

Smanji confidence threshold u skripti:

```python
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.3  # Smanji sa 0.5 na 0.3
)
```

---

## Dodatna dokumentacija

- **Detaljno**: Vidi `README_ZADATAK1.md`
- **Pomoćne funkcije**: Vidi `src/utils_keypoint_mapping.py`

---

## Kontakt

Za pitanja i problem:
- GitHub Issues: https://github.com/KxHartl/interakcija-covjeka-i-robota/issues
- Autor: Krešimir Hartl
