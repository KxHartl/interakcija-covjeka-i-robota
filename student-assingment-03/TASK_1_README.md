# Zadatak 1: Usporedba Ground Truth i MediaPipe Podataka

## Pregled

Ovaj folder sadrži Python skripte za analizu podataka iz HPE-24 dataseta, usporedbu ground truth (BIWI) pose podataka sa MediaPipe FaceMesh detektom.

**Data folder:** `student-assingment-03/data/raw/hpe/24`

---

## Dostupne Skripte

### 1. **00_task_1_main.py** - Glavna Analiza

**Svrha:** 
- Učitavanje ground truth pozicije i rotacije glave
- Pokretanje MediaPipe FaceMesh na RGB slikama
- Eksportiraj rezultate u JSON format

**Pokretanje:**
```bash
python student-assingment-03/src/00_task_1_main.py
```

**Izlaz:**
- `./results/task_1_analysis.json` - Detaljni rezultati analize po frameovima

**Prilagodljivi parametri** (u `__main__` dijelu):
- `frame_start` - Početni frame ID (default: 100)
- `frame_end` - Završni frame ID (default: 120)

---

### 2. **01_task_1_visualization.py** - Vizualizacija

**Svrha:**
- Crtanje 3D koordinatnih osi (ground truth)
- Prikaz MediaPipe keypointa
- Bounding box oko detektiranog lica
- Spremi vizualizaciju kao PNG sliku

**Pokretanje:**
```bash
python student-assingment-03/src/01_task_1_visualization.py
```

**Izlaz:**
- `./results/task_1_visualization.png` - Vizualizirana slika sa oba seta podataka

**Prilagodljivi parametri** (u `__main__` dijelu):
- `frame_id` - Frame koji želiš prikazati (default: 100)

---

### 3. **02_task_1_statistics.py** - Statistička Analiza

**Svrha:**
- Izračun statističkih mjera (mean, std, min, max)
- Analiza Euler kutova (pitch, yaw, roll)
- Analiza translacije (x, y, z)
- Crtanje distribucijskih grafika

**Pokretanje:**
```bash
python student-assingment-03/src/02_task_1_statistics.py
```

**Izlaz:**
- `./results/task_1_statistics.json` - Statistički rezultati
- `./results/euler_distributions.png` - Grafiki distribucija kutova

**Prilagodljivi parametri** (u `__main__` dijelu):
- `frame_start` - Početni frame (default: 100)
- `frame_end` - Završni frame (default: 120)

---

## Instalacija Ovisnosti

```bash
pip install -r student-assingment-03/requirements.txt
```

**Potrebne biblioteke:**
- `opencv-python` - Obrada slika
- `numpy` - Numerika
- `mediapipe` - FaceMesh detektor
- `scipy` - Naučne funkcije
- `pandas` - Obrada podataka
- `matplotlib` (opciono) - Grafici
- `seaborn` (opciono) - Napredni grafiki
- `pillow` - Obrada slika

---

## Struktura Podataka

### Ulazni Podaci (HPE-24)

Svaki frame u `/data/raw/hpe/24/` sadrži:

```
frame_XXXXX_rgb.png        RGB slika (640x480)
frame_XXXXX_pose.txt       Ground truth pose (rotacija + translacija)
frame_XXXXX_depth.bin      Depth mapa (opciono)
```

### Format Pose Datoteke

```
rotacija_11 rotacija_12 rotacija_13
rotacija_21 rotacija_22 rotacija_23
rotacija_31 rotacija_32 rotacija_33
translacija_x translacija_y translacija_z
```

- **Rotacija:** 3x3 rotacijska matrica
- **Translacija:** u milimetrima (mm)

---

## Output Formati

### JSON Analiza (task_1_analysis.json)

```json
{
  "timestamp": "2025-12-11T08:46:43.000000",
  "total_frames": 21,
  "data_folder": "./student-assingment-03/data/raw/hpe/24",
  "results": [
    {
      "frame_id": 100,
      "status": "success",
      "ground_truth": {
        "rotation_matrix": [[...], [...], [...]],
        "translation_m": [x, y, z],
        "euler_angles_deg": [pitch, yaw, roll]
      },
      "mediapipe": {
        "detected": true,
        "landmark_count": 468
      }
    }
  ]
}
```

### Statistička Analiza (task_1_statistics.json)

```json
{
  "frame_count": 21,
  "data_folder": "./student-assingment-03/data/raw/hpe/24",
  "euler": {
    "pitch": { "mean": -10.45, "std": 8.23, "min": -25.67, "max": 5.89 },
    "yaw": { "mean": 2.34, "std": 12.56, "min": -15.23, "max": 28.45 },
    "roll": { "mean": 1.23, "std": 3.45, "min": -8.56, "max": 9.87 }
  },
  "translation_m": {
    "x": { "mean": 0.023, "std": 0.045, ... },
    "y": { "mean": 0.145, "std": 0.032, ... },
    "z": { "mean": 0.567, "std": 0.078, ... }
  }
}
```

---

## Kalibracija Kamere

Kalibracijska matrica za BIWI dataset:

```python
K = np.array([
    [575.816,   0.0,    320.0],
    [0.0,     575.816, 240.0],
    [0.0,       0.0,      1.0]
], dtype=np.float32)
```

- **Focal length:** 575.816 px
- **Principal point:** (320, 240)

---

## Legenda Vizualizacije

U `01_task_1_visualization.py` je prikazano:

- **Žuta točka** - Ishodite (glava) po ground truth
- **Crvena linija (X os)** - Ground truth X os
- **Zelena linija (Y os)** - Ground truth Y os
- **Plava linija (Z os)** - Ground truth Z os
- **Cyan točke** - MediaPipe keypointi (468 lica)
- **Zeleni pravokutnik** - Bounding box oko lica

---

## Napomene

### Framea
Framea su označeni sa brojevima 100-199. Nije sve framee dostupni - skripte ignoriraju nedostajuće.

### Euler Kutovi
Kutovi se ispisuju kao (pitch, yaw, roll) u stupnjeva:
- **Pitch** - Rotacija oko X osi (glave gore/dolje)
- **Yaw** - Rotacija oko Y osi (glave lijevo/desno)
- **Roll** - Rotacija oko Z osi (glave lijevo/desno)

### MediaPipe Detekcija
MediaPipe koristi:
- `static_image_mode=True` - za procesiranje pojedinačnih slika
- `max_num_faces=1` - pretpostavljamo samo jedno lice
- `min_detection_confidence=0.6` - minimalna pouzdanost 60%

---

## Troubleshooting

**Problem:** Path ne postoji
```
⚠️  Putanja ne postoji: ./student-assingment-03/data/raw/hpe/24
```
**Rješenje:** Prilagodi `data_root` putanju u `__main__` dijelu prema tvojoj strukturi direktorija.

**Problem:** Frame ne postoji
```
[Frame 00100] Status: file_not_found
```
**Rješenje:** Promijeni `frame_id` ili `frame_start`/`frame_end` na dostupne framee.

**Problem:** Matplotlib nije dostupan
```
⚠️  Matplotlib nije dostupan
```
**Rješenje:** Instaliraj matplotlib: `pip install matplotlib`

---

## Dodatni Resursi

- [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [BIWI Head Pose Dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/3dpose/)
- [Euler Angles](https://en.wikipedia.org/wiki/Euler_angles)

---

## Autor i Datum

**Student:** [Tvoje ime]
**Datum:** 2025-12-11
**Dataset:** HPE-24 (BIWI Head Pose)

---
